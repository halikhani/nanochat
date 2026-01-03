""" Reinforcement learning on GSM8k via GRPO

we actually end up with something a lot simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use GAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import os
import itertools
import re
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# RL hyperparams
run = "dummy" # wandb run name
source = "sft" # mid|sft
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
dtype = "bfloat16"
device_batch_size = 8 # no forward pass above this to avoid OOM
examples_per_step = 16 # in total and across all ranks (note: examples, not samples/completions!) #NOTE: get back to this
num_samples = 16 # number of samples per example (/question)
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: try None? (from nanochat repo)
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # how many epochs of gsm8k to train on
save_every = 60 # every how many steps to save the model
eval_every = 60 # every how many steps to evaluate the model for val pass@k 
eval_examples = 400 # number of examples used for evaluating pass@k
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# init compute/precision
ddp, ddp_rank, ddp_loca_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)
# what is autocast_ctx? 
# “When I run ops inside this context, automatically use a lower-precision dtype (when safe) on this device 
# to make things faster and more memory-efficient.”

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# init model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
engine = Engine(model, tokenizer) # for sampling rollouts


# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_specia("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank for different examples
    for example_idx in itertools.cycle(rank_indices): # cycle through the examples for each rank indefinitely

        # first get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (keeping the |assistant_end| but deleting everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # generate num_samples samples using batched generation, using a loop to avoid OOM
        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # to avoid OOM
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            with autocast_ctx:
                generated_tokens_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed, # must make sure to change the seed for each sampling step   
                )
            generated_token_sequences.extend(generated_tokens_sequences_batch)
            masks.extend(masks_batch)

        # calculate the reward for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # only getting the generated tokens after the prompt
            generated_tokens = sample_tokens[prefix_length:]
            # decode the generated tokens
            generated_text = tokenizer.decode(generated_tokens)
            # calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # pad the sequences to the same length
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [ seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # stack up into torch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # generate autoregressive inputs and targets to the transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # avoiding modification
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index (ignore index when mask is )
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # calculating the advantages by subtracting the mean (instead of the z-score (x-mu)/sigma))
        mu = rewards.mean()
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages

    
# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine, max_examples=None, num_samples=1, max_completion_tokens=256, temperature=1.0, top_k=50):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    
    This pattern is useful for long-running evaluations where you want to process results 
    incrementally rather than waiting for everything to finish.
    """

    # max_examples is the maximum number of examples to evaluate, or the total number of examples if not specified
    # num_samples is the number of samples to generate for each example

    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # generate k samples using batched generation inside the engine
        assert num_samples <= device_batch_size, # usually this is true. we can add a loop if not to avoid OOM
        generated_tokens_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # check each sample for correctness
        outcomes = []
        for sample_tokens in generated_tokens_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct,
            })
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record
        # Each iteration of the loop:
        # Processes one example (distributed across ranks)
        # Generates samples
        # Evaluates correctness
        # Yields one record dictionary

        # Example usage pattern:
        # records_iter = run_gsm8k_eval(val_task, tokenizer, engine, ...)
        # for record in records_iter:
        # Process each record as it arrives
        # Can break early if needed
       


    




