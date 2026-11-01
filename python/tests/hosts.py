"""What this host is, for the tests' skip conditions.

ARM64 / AMD64: the machine's architecture (tests of one backend's code generation need theirs).
RUNS: cimo.compile works here, so tests that only run C need nothing more (arm64 Linux, macOS,
Windows; x86-64 Linux and Windows).
"""

import platform
import sys

MACHINE = platform.machine().lower()
ARM64 = MACHINE in ('arm64', 'aarch64')
AMD64 = MACHINE in ('x86_64', 'amd64', 'x64')
RUNS = ARM64 or (AMD64 and sys.platform != 'darwin')
