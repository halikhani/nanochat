"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example run:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 scripts.chat_eval -a ARC-Easy
"""

import argparse
from functools import partial
from contextlib import nullcontext

from hamidra.KVC.tutorials.nanochat.nanochat.dataset import num
import torch
import torch.distributed as dist

from nanochat.common import compute_init, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


# ------------------------------------------------------------
# generative evaluation loop, one problem at a time, sample, evaluate

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # tokenize the prompt 
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # get the completion
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # decode the completion as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode((result_tokens[prefix_length:])) for result_tokens in results]
        # evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # update stats
        total += 1
        num_passed += int(passed)

        # logging (overwrite same line)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Return the accuracy
    return num_passed/total

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.
def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    pass