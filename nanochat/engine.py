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
        return choice
        # whats rng? random number generator

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens if current_tokens is not None else [] # current tokens in the row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we're inside a Python block
        self.python_expr_tokens = [] # List of tokens that are part of the Python expression
        self.completed = False #  Whether this row has completed generation
        
class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """ same as generate() in gpt.py, but does single prefill and then clones the KV cache"""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting a list of token ids"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # get the special tokens we need to coordinate the tool use state machines
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start, python_end = get_special("<|python_start|>"), get_special("<|python_end|>")
        output_start, output_end = get_special("<|output_start|>"), get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1. Run a batch 1 prefill of the prompt token
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_heads,
            "head_dim": m.n_embd // m.n_kv_heads,
            "num_layers": m.n_layer,
        }
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_model_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :] # (B, 1, vocab_size) for the last token in the prompt, B is the batch size which is 1 in this case
        next_ids = sample_next_token(logits, rng, temperature, top_k) # (B, 1), in which batch size is 1
        sampled_tokens = next_ids[:, 0].tolist() # (1,)

        # 2. Replicate KV cache for each sample/row
        # why? because we want to generate multiple samples in parallel from the same prefill state

        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else m.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # free memory

        # 3. Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4. Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # stop condition: we've reached the max number of generated tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # stop condition: all rows have completed generation
            if all(state.completed for state in row_states):
                break

            # get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                sampled_tokens = [sampled_tokens[0]] * num_samples # broadcast the first token to all rows
                # TODO: we should sample first token for each row separately
                first_iteration = False
            else:
                # forwad the model and get the next token for each row
                logits = self.model.forward(ids, kv_cache=kv_cache_decode) # (B, T, vocab_size)
                logits = logits[:, -1, :] # (B, 1, vocab_size) for the last token in the sequence
                next_ids = sample_next_token(logits, rng, temperature, top_k) # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist() # (B,)

            # process each row: choose the next token, update state, optional tool use
            token_columns = [] # next token ids for each row
            token_masks = [] # contains mask (was it sampeld (1) or forced (0)) along each row
            for i, state in enumerate(row_states):
                # select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_columns.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_toknes = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.decode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
            
            # yield the token column
            yield token_columns, token_masks
            # how upper line works? token_columns is a list of token ids, token_masks is a list of mask values (1 for sampled, 0 for forced)
            # so we are yielding the token ids and the mask values for each row  
            num_generated += 1
            # prepare ids for next iteration
            ids = torch.tensor(token_columns, dtype=torch.long, device=device).unsqueeze(1) # (B, 1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences
        Returns a list of token sequences (list of list of ints)
        Terminal tokens are not included (e.g., <|assistant_end|> or <|bos|>)
        """

        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_mask in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_mask)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # stop of all rows are completed
            if all(completed):
                break
        return results, masks

if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """

    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # load the model and the tokenizer
    model, tokenizer, meta = load_model("base", device=device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparams
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula for water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    # whats autocast_ctx? it is a context manager that allows us to use mixed precision training
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.4f} seconds")
    reference_ids = generated_tokens
    # generate tokens with engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # run in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_mask in stream:
            token = token_column[0] # only print out the first row
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.4f} seconds")
    engine_ids = generated_tokens
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != engine_ids[i]:
            print(f"Mismatch at index {i}: {reference_ids[i]} != {engine_ids[i]}")
            break
    print(f"Match: {reference_ids == engine_ids}")
