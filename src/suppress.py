import os, sys
from contextlib import contextmanager

@contextmanager
def suppress_c_output():
    # 1) Flush Python buffers
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

    # 2) Open the null device and save original FDs
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    # 3) Redirect both stdout (fd=1) and stderr (fd=2) to /dev/null
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)

    try:
        yield
    finally:
        # 4) Restore the original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        # 5) Clean up
        os.close(null_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

