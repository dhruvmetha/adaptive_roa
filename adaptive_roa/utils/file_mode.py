"""The mode a plain open() would give a new file, for temp files made by mkstemp.

tempfile.mkstemp always creates 0600 and os.replace keeps the temp file's mode, so an
atomic write ignores the umask. Run outputs live in a tree shared with login-bekris
under umask 002; chmod the temp file to new_file_mode() before os.replace.
"""
from __future__ import annotations

import os


def current_umask() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except OSError:
        pass
    # No /proc: os.umask can only be read by setting it, so put it straight back.
    old = os.umask(0o022)
    os.umask(old)
    return old


def new_file_mode() -> int:
    return 0o666 & ~current_umask()
