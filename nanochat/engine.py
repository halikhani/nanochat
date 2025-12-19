"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from contextlib import nullcontext

# ------------------------------------------------------------
# calculator tool helpers 
@contextmanager
def timeout(duration, formula):
    # when alarm fires, raise an exception
    # register a handler, schedule an alarm, yield control to with body, cancel alarm after exit
    # guarantees no infinite loops or hung python execution
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")
    # next line: “When a SIGALRM signal happens, run timeout_handler.”
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

# Start a timer

# Run some code

# If the code takes too long → raise an exception

# When done → turn the timer off
def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                # next line: 
                # Important safety detail:
                # __builtins__ is empty
                # No open, no import, no exec, no eval
                # Only pure expressions like 2+3, 1/7, etc.
                return eval(formula,  {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # remove commas from numbers
    expr = expr.replace(",", "")
    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None
    
    # disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # only allow .count() for now
    if '.count(' not in expr:
        return None

    # evaluate the expression with timeout
    return eval_with_timeout(expr)

# ------------------------------------------------------------

