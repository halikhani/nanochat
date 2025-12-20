"""
Sandboxed execution utilities for running Python code that comes out of an LLM.
Adapted from OpenAI HumanEval code:
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

What is covered:
- Each execution runs in its own process (can be killed if it hangs or crashes)
- Execution is limited by a timeout to stop infinite loops
- Memory limits are enforced by default (256MB)
- stdout and stderr are captured and returned
- Code runs in a temporary directory that is deleted afterwards
- Dangerous functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)

What is not covered:
- Not a true security sandbox
- Network access is not blocked (e.g. sockets could be opened)
- Python's dynamic features (e.g. ctypes) could bypass restrictions
- No kernel-level isolation (no seccomp, no containers, no virtualization)

Overall this sandbox is good for evaluation of generated code and protects against
accidental destructive behavior, but it is not safe against malicious adversarial code.
"""

import contextlib # structured setup/teardown
import faulthandler
import io
import multiprocessing
import os
import platform
import signal # timeout signals
import tempfile # temporary files
from dataclasses import dataclass
from typing import Optional

# ----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """ Result of executing python code in a sandbox."""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False

    def __repr__(self) -> str:
        parts = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out after {seconds} seconds")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)

    try:
        yield
        # control returns to the 'with' block
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # always runs, even on timeout signal.setitimer(signal.ITIMER_REAL, 0): cancels the timer

@contextlib.contextmanager
def capture_io():
    """Capture stdout and stderr, and disable stdin."""
    # Used in the sandbox to:
    # Capture all output for inspection
    # Prevent code from reading from stdin (security/isolation)
    # Return captured output in the ExecutionResult
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    # stdout → buffer
    # stderr → buffer
    # stdin → write-only, reading raises error
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_temp_dir():
    with tempfile.TemporaryDirectory() as dir_name:
        with os.chdir(dir_name):
            # change dir to the temporary directory
            yield dir_name

class TimeoutException(Exception):
    pass

