"""Start a Python helper process without fork().

Every helper this server starts (the detector worker, a survey job, a
GhostTrace run) used subprocess.Popen with cwd= and, for the job worker,
start_new_session=True. On macOS those two arguments force CPython onto the
fork()+exec() path. Between fork and exec the child runs every pthread_atfork
handler registered in the parent, and this parent accumulates a lot of native
code over a session: faiss's OpenMP, OpenCV's ffmpeg/AVFoundation stack from
single-tile verification, pyproj, shapely, CoreFoundation. After enough of
that, one of those handlers crashed the child before exec: the job worker
died with SIGSEGV, still in the parent's process group, with an empty log,
and the same server that had run three surveys an hour earlier failed every
one after that. Nothing in the worker itself was at fault; a fresh server ran
the identical job.

posix_spawn() has no such window: macOS implements it in the kernel and no
parent-side handler runs. CPython uses it inside Popen when, and only when,
there is no cwd, no start_new_session, no preexec_fn, no pass_fds, env is
None and close_fds is False. So this module meets those conditions and moves
the two things they replaced into the child itself, in a one-line bootstrap
that runs before anything else is imported:

    os.setsid()          the worker leads its own session, so a kill of the
                         group still reaches its own children
    os.chdir(ROOT)       and ROOT on sys.path, so `-m backend.x` semantics hold

close_fds=False is safe here: every descriptor CPython creates is
non-inheritable (PEP 446), and posix_spawn honours CLOEXEC, so the child sees
its own three standard streams and nothing of the parent's sockets or pipes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from . import config


def _bootstrap(target: str, *, module: bool, setsid: bool) -> str:
    root = str(config.ROOT)
    run = (f"runpy.run_module({target!r}, run_name='__main__', alter_sys=True)" if module
           else f"runpy.run_path({target!r}, run_name='__main__')")
    return ("import os, sys, runpy; "
            + ("os.setsid(); " if setsid else "")
            + f"os.chdir({root!r}); sys.path.insert(0, {root!r}); " + run)


def spawn_python(target: str | Path, args: list[str] = (), *, module: bool = True,
                 setsid: bool = False, **popen_kwargs: Any) -> subprocess.Popen:
    """Popen for `python -m target args` (or a script path), via posix_spawn.

    `popen_kwargs` may carry stdin/stdout/stderr/text/bufsize and the like.
    cwd, start_new_session, env, close_fds, pass_fds and preexec_fn are not
    accepted: each one would silently put CPython back onto fork().
    """
    forbidden = {"cwd", "start_new_session", "env", "close_fds", "pass_fds", "preexec_fn"}
    bad = forbidden & set(popen_kwargs)
    if bad:
        raise TypeError(f"spawn_python does not take {sorted(bad)}; see backend/procs.py")
    boot = _bootstrap(str(target), module=module, setsid=setsid)
    return subprocess.Popen([sys.executable, "-c", boot, *args], close_fds=False, **popen_kwargs)
