"""A one-off differential fuzzing campaign on an AWS EC2 arm64 instance, for runs too long or
too heavy for a workstation (CI fuzzing is the nightly GitHub Actions job, .github/workflows/
fuzz.yml): Csmith, then YARPGen, against gcc -O0 (fuzz_csmith.py, fuzz_yarpgen.py), for a time
budget. Failures come back to tests/fuzz_failures/aws-<time>/.

    python tests/fuzz_aws.py [--minutes 60] [--type c7g.4xlarge] [--commit HEAD]

Needs the AWS CLI with credentials (region from its config) and ssh/scp. The commit must be
pushed: the instance clones it from GitHub. Everything the run creates is tagged
Name=fariz-fuzz and removed at the end (instance, security group, key pair), also on errors
and Ctrl-C. As a backstop the instance shuts itself down, which terminates it, 20 minutes
after its budget, so a dead driver cannot leave it running. `--cleanup` terminates any
leftover fariz-fuzz instances and deletes leftover groups and keys.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = 'https://github.com/farizrahman4u/cimo'
AMI_PARAM = '/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id'
TAG = 'fariz-fuzz'  # the account's naming: a fariz- prefix, and no project name in any resource

USER_DATA = r'''#!/bin/bash
shutdown -h +{hard}  # backstop: shutdown terminates the instance
exec > /var/log/fariz-fuzz.log 2>&1
set -x
apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q git gcc g++ cmake m4 python3 python3-pytest
sudo -u ubuntu bash <<'EOF'
set -x
cd ~
mkdir -p code
git clone -q https://github.com/csmith-project/csmith code/csmith
cmake -S code/csmith -B code/csmith/build -DCMAKE_BUILD_TYPE=Release
cmake --build code/csmith/build -j$(nproc)
git clone -q https://github.com/intel/yarpgen code/yarpgen
cmake -S code/yarpgen -B code/yarpgen/build -DCMAKE_BUILD_TYPE=Release
cmake --build code/yarpgen/build -j$(nproc)
git clone -q {repo} cimo
cd cimo && git checkout -q {commit} && cd python
t=$(date +%s)  # a fresh range per run (the nightly CI job seeds from the day)
python3 tests/fuzz_csmith.py -n 1000000 --seed $(( t * 10000 )) --minutes {csmith} \
    --timeout 600 > ~/fuzz.log 2>&1
python3 tests/fuzz_yarpgen.py -n 1000000 --seed $(( t * 1000 )) --minutes {yarpgen} \
    --timeout 900 >> ~/fuzz.log 2>&1
mkdir -p tests/fuzz_failures
cp ~/fuzz.log tests/fuzz_failures/
tar -czf ~/results.tar.gz -C tests fuzz_failures
touch ~/done
EOF
'''


def aws(*args):
    """Runs an AWS CLI command; its JSON output, parsed (None if empty)."""
    p = subprocess.run(['aws'] + list(args) + ['--output', 'json'], capture_output=True,
                       text=True)
    if p.returncode:
        raise RuntimeError('aws %s: %s' % (' '.join(args[:2]), p.stderr.strip()))
    return json.loads(p.stdout) if p.stdout.strip() else None


def ssh(ip, key, cmd, **kw):
    return subprocess.run(['ssh', '-i', key, '-o', 'StrictHostKeyChecking=no',
                           '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=10',
                           '-o', 'LogLevel=ERROR', 'ubuntu@' + ip, cmd], **kw)


def leftovers():
    """Instance ids of fariz-fuzz instances that are not terminated."""
    r = aws('ec2', 'describe-instances', '--filters', 'Name=tag:Name,Values=' + TAG,
            'Name=instance-state-name,Values=pending,running,stopping,stopped')
    return [i['InstanceId'] for res in r['Reservations'] for i in res['Instances']]


def cleanup(instance=None, group=None, key=None):
    """Terminates and deletes what a run created (or, with no arguments, every leftover)."""
    ids = [instance] if instance else leftovers()
    if ids:
        aws('ec2', 'terminate-instances', '--instance-ids', *ids)
        subprocess.run(['aws', 'ec2', 'wait', 'instance-terminated', '--instance-ids'] + ids)
        print('terminated', ' '.join(ids))
    if group is None and instance is None:
        r = aws('ec2', 'describe-security-groups', '--filters', 'Name=tag:Name,Values=' + TAG)
        groups = [g['GroupId'] for g in r['SecurityGroups']]
        r = aws('ec2', 'describe-key-pairs', '--filters', 'Name=tag:Name,Values=' + TAG)
        keys = [k['KeyName'] for k in r['KeyPairs']]
    else:
        groups = [group] if group else []
        keys = [key] if key else []
    for args in [('delete-security-group', '--group-id', g) for g in groups] + \
            [('delete-key-pair', '--key-name', k) for k in keys]:
        try:
            aws('ec2', *args)
        except RuntimeError as e:  # already gone (a retried cleanup)
            if 'NotFound' not in str(e):
                raise
    if groups or keys:
        print('deleted', ' '.join(groups + keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--minutes', type=float, default=60, help='fuzzing time (split 2:1)')
    ap.add_argument('--type', default='c7g.4xlarge', help='instance type (arm64)')
    ap.add_argument('--commit', default='HEAD')
    ap.add_argument('--cleanup', action='store_true', help='remove leftovers and exit')
    a = ap.parse_args()
    if a.cleanup:
        cleanup()
        return 0
    commit = subprocess.check_output(['git', 'rev-parse', a.commit], cwd=HERE, text=True).strip()
    if subprocess.run(['git', 'branch', '-r', '--contains', commit], cwd=HERE,
                      capture_output=True, text=True).stdout.strip() == '':
        sys.exit('commit %s is not pushed' % commit[:10])
    stamp = time.strftime('%Y%m%d-%H%M%S')
    name = '%s-%s' % (TAG, stamp)
    tags = 'ResourceType=%s,Tags=[{Key=Name,Value=' + TAG + '}]'
    tmp = tempfile.mkdtemp(prefix='cimo-aws-')
    key = os.path.join(tmp, 'key.pem')
    instance = group = keyname = ip = None
    fetched = False
    out = os.path.join(HERE, 'fuzz_failures', 'aws-' + stamp)
    try:
        r = aws('ec2', 'create-key-pair', '--key-name', name, '--key-type', 'ed25519',
                '--tag-specifications', tags % 'key-pair')
        keyname = name
        with open(key, 'w') as f:
            f.write(r['KeyMaterial'])
        os.chmod(key, 0o600)
        myip = urllib.request.urlopen('https://checkip.amazonaws.com').read().decode().strip()
        vpc = aws('ec2', 'describe-vpcs', '--filters', 'Name=isDefault,Values=true')['Vpcs'][0]
        group = aws('ec2', 'create-security-group', '--group-name', name, '--description',
                    'fuzzing run (ssh from the driver)', '--vpc-id', vpc['VpcId'],
                    '--tag-specifications', tags % 'security-group')['GroupId']
        aws('ec2', 'authorize-security-group-ingress', '--group-id', group, '--protocol', 'tcp',
            '--port', '22', '--cidr', myip + '/32')
        ami = aws('ssm', 'get-parameter', '--name', AMI_PARAM)['Parameter']['Value']
        csm, yg = a.minutes * 2 / 3, a.minutes / 3
        ud = USER_DATA.format(hard=int(a.minutes + 40), repo=REPO, commit=commit, csmith=csm,
                              yarpgen=yg)
        r = aws('ec2', 'run-instances', '--image-id', ami, '--instance-type', a.type,
                '--key-name', keyname, '--security-group-ids', group,
                '--instance-initiated-shutdown-behavior', 'terminate',
                '--block-device-mappings',
                '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3",'
                '"DeleteOnTermination":true}}]',
                '--user-data', ud, '--tag-specifications', tags % 'instance')
        instance = r['Instances'][0]['InstanceId']
        print('%s: %s (%s), commit %s, %.0f minutes' % (name, instance, a.type, commit[:10],
                                                         a.minutes), flush=True)
        subprocess.check_call(['aws', 'ec2', 'wait', 'instance-running', '--instance-ids',
                               instance])
        ip = aws('ec2', 'describe-instances', '--instance-ids', instance)[
            'Reservations'][0]['Instances'][0]['PublicIpAddress']
        t0 = time.time()
        errors = 0  # consecutive failed status checks (a network blip must not end the run)
        while True:
            time.sleep(60)
            p = ssh(ip, key, 'test -f ~/done && echo done; tail -n 2 ~/fuzz.log 2>/dev/null',
                    capture_output=True, text=True)
            if 'done' in p.stdout.split('\n', 1)[0]:
                break
            try:
                state = aws('ec2', 'describe-instances', '--instance-ids', instance)[
                    'Reservations'][0]['Instances'][0]['State']['Name']
                errors = 0
            except RuntimeError as e:
                errors += 1
                print('  status check failed (%d): %s' % (errors, e), flush=True)
                if errors >= 10:
                    raise
                continue
            if state not in ('pending', 'running'):
                raise RuntimeError('instance is %s before finishing' % state)
            print('  %3.0f min: %s' % ((time.time() - t0) / 60,
                                       p.stdout.strip().replace('\n', ' | ')[-200:]), flush=True)
        os.makedirs(out)
        subprocess.check_call(['scp', '-i', key, '-o', 'StrictHostKeyChecking=no', '-o',
                               'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR',
                               'ubuntu@%s:results.tar.gz' % ip, tmp])
        with tarfile.open(os.path.join(tmp, 'results.tar.gz')) as t:
            t.extractall(out, filter='data')
        with open(os.path.join(out, 'fuzz_failures', 'fuzz.log')) as f:
            print(f.read())
        print('results in', out)
        fetched = True
    finally:
        if ip and not fetched:  # salvage what there is before the instance goes
            p = ssh(ip, key, 'cd ~/cimo/python && mkdir -p tests/fuzz_failures && '
                    'cp ~/fuzz.log tests/fuzz_failures/ 2>/dev/null; '
                    'tar -czf ~/partial.tar.gz -C tests fuzz_failures', capture_output=True)
            os.makedirs(out, exist_ok=True)
            if p.returncode == 0 and subprocess.run(
                    ['scp', '-i', key, '-o', 'StrictHostKeyChecking=no', '-o',
                     'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR',
                     'ubuntu@%s:partial.tar.gz' % ip, tmp]).returncode == 0:
                with tarfile.open(os.path.join(tmp, 'partial.tar.gz')) as t:
                    t.extractall(out, filter='data')
                print('partial results in', out)
        for attempt in range(10):  # cleanup must not give up on a network blip either
            try:
                cleanup(instance, group, keyname) if instance or group or keyname else None
                left = leftovers()
                break
            except RuntimeError as e:
                print('cleanup failed (%s), retrying' % e, flush=True)
                time.sleep(30)
        else:
            left = ['(unknown: run `python tests/fuzz_aws.py --cleanup`)']
        shutil.rmtree(tmp, ignore_errors=True)
        if left:
            print('WARNING: fariz-fuzz instances still running:', ' '.join(left))
    return 0


if __name__ == '__main__':
    sys.exit(main())
