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

class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # each of K/V is of shape (batch_size, num_heads, seq_len, head_dim)
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in the time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """

        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty cache"
        assert other.kv_cache is not None, "Cannot prefill with an empty cache"
        
        # extract dims explicitly
        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = self.kv_shape
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = other.kv_shape

        # validate dims
        assert self_layers == other_layers, "Cannot prefill with a cache of different number of layers"
        assert self_kv == other_kv, "Cannot prefill with a cache of different number of key/value pairs"
        assert self_heads == other_heads, "Cannot prefill with a cache of different number of heads"
        assert self_head_dim == other_head_dim, "Cannot prefill with a cache of different head dimension"

        # batch size can be expanded, other should be 1, self can be larger
        assert self_batch == other_batch or other_batch == 1, f"Batch size mismatch: {self_batch} vs {other_batch} (other must be 1 or equal)"

        # Sequence length: self must be longer than other
        assert self_seq >= other_seq, f"Sequence length mismatch: {self_seq} vs {other_seq} (self must be longer or equal)"

        # 2) Initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[..., :other_seq, :] = other.kv_cache
        # 4) update the position
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # lazy init the cache here since we need to know the dtype and device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        
        # insert the new k, v into the cache and return the full view so far
        B, H, T_add, D = k.size() # k shape: (batch, num_heads, tokens_to_add, head_dim)
        t0, t1 = self.pos, self.pos + T_add

        # dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4): # fifth dim
            t_needed = t1 + 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            # other way of doing this:
            # t_needed = (t1 + 1023) // 1024 * 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape

        # insert the new k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        # update pos after last layer of the Transformer inserts
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view

# -----------------------------------------------------------------------------

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0, "Temperature must be non-negative"
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True) # greedy sampling, returning the index of the highest logit

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        # whats rng? random number generator

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens if current_tokens is not None else [] # current tokens in the row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we're inside a Python block
        self.python_expr_toknes = [] # List of tokens that are part of the Python expression
        self.completed = False #  Whether this row has completed generation
        