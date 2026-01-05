""" 
New and upgraded chat CLI 

for now intended only for single GPU inference.
python -m scripts.chat_cli -i mid
"""

import argparse
from enum import auto
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine, ddp_world_size
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# init model and tokenizer

device_type = autodetect_device_type() if args.device_type == '' else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# create the engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # get the prompt from the command line
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
    
    # handle special commands
    if user_input.lower() in ["quit", "exit"]:
        print("\nGoodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # add user message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("Assistant: ", end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    # add assistant message to the conversation
    conversation_tokens.extend(response_tokens)
    
    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
    