"""The site's agent-facing index stays in step with the docs."""
import json
import os
import re

DOCS = os.path.join(os.path.dirname(__file__), '..', '..', 'docs')


def test_llms_txt_lists_every_docs_page():
    with open(os.path.join(DOCS, 'docs', 'nav.json')) as f:
        pages = [p[0] for part in json.load(f)['parts'] for p in part['pages']]
    with open(os.path.join(DOCS, 'llms.txt')) as f:
        linked = re.findall(r'https://www\.cimo\.dev/docs/content/([\w-]+)\.md', f.read())
    assert linked == pages
    for page in linked:
        assert os.path.exists(os.path.join(DOCS, 'docs', 'content', page + '.md'))


TAG = re.compile(r'''((?:src|href)="([\w./-]+\.(?:js|css))\?v=)([0-9a-f]+)"''')


def _tags(fix=False):
    """The ?v= on the site's script and stylesheet tags: the first 8 hex digits of the file's SHA-256, so a browser
    fetches a changed file instead of its cached copy. Returns the stale ones; with fix, rewrites them."""
    import hashlib
    stale = []
    for page in ('index.html', os.path.join('docs', 'index.html')):
        path = os.path.join(DOCS, page)
        with open(path, newline='') as f:
            text = f.read()

        def sub(m):
            target = os.path.normpath(os.path.join(os.path.dirname(path), m.group(2)))
            with open(target, 'rb') as f:
                # as deployed: a Windows checkout has CRLF line ends, the site is built from LF
                want = hashlib.sha256(f.read().replace(b'\r\n', b'\n')).hexdigest()[:8]
            if m.group(3) != want:
                stale.append('%s: %s?v=%s, the file is %s' % (page, m.group(2), m.group(3), want))
            return m.group(1) + want + '"'
        new = TAG.sub(sub, text)
        if fix and new != text:
            with open(path, 'w', newline='') as f:
                f.write(new)
    return stale


def test_script_versions_match_the_files():
    """A demo script changed without its ?v= kept the old copy in browsers, next to a newly deployed cimo.py it no
    longer matched (the attractor demo's "keep its signature"). python tests/test_site.py rewrites them."""
    stale = _tags()
    assert not stale, 'stale ?v= tags (python tests/test_site.py fixes them):\n' + '\n'.join(stale)


if __name__ == '__main__':
    for line in _tags(fix=True):
        print('updated', line)
