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
    pass