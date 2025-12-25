"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""


import random
from jinja2 import Template
import torch
import torch.distributed as dist


# ------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """ Render complete prompts for multiple choice tasks. """
    template_str = """
    {%- for example in fewshot_examples -%}
    {{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

    {% endfor -%}
    {{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()

    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item.choices]
    # So if choices = ["A", "B", "C"], you get 3 prompts:
    # prompt ending in “... delimiter A”
    # prompt ending in “... delimiter B”
    # prompt ending in “... delimiter C”
    # run all 3, score only the answer region, choose lowest loss.
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
    {%- for example in fewshot_examples -%}
    {{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

    {% endfor -%}
    {{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context) for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """

    template_str = """
    {%- for example in fewshot_examples -%}
    {{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

    {% endfor -%}
    {{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }

    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, we need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """

    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(min_len - 1, -1, -1)
    }[direction]

    # finding the first position where the sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i # we can return i because of 0-indexing
    return min_len


def stack_sequences(tokens, pad_token_id):
    """ Stack up a list of token sequences, pad to the longest on the right. """
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long) # making a tensor of all pad tokens with size bsz x seq_len
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # in multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction="right")
    start_indices = [answer_start_idx] * len(prompts) # all the same because the context is the same
    end_indices = [len(x) for x in tokens] # all the different because the continuations are different
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # in schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out start and end of each context 
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx] # why in list? because we need to return a list of lists and be consistent with the other functions


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # roll the tensor to left by 1 position to get the autoregressive target ids

    # input_ids = [A, B, C, D]  # Original sequence
    # After torch.roll(input_ids, shifts=-1, dims=1):
    # target_ids = [B, C, D, ?]  # Shifted left by 1

    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len, -1),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = torch.argmax(outputs, dim=-1)
    return losses, predictions



@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """ Evaluate a single example, return true if correct, false otherwise. """
    item = data[idx]
    task_type = task_meta['type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

     # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # render prompts and batch sequences based on the task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_lm(tokenizer, prompts)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # some models cant forward sequences beyong a certain length (like GPT-2)
    # in these cases, we have to truncate sequences to max length and adjust the indices

    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_indices, new_end_indices = [], [], []
        for t, s, e in zip(tokens, start_indices, end_indices):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # taking the last max_tokens tokens
                new_start_indices.append(s - num_to_crop)
                new_end_indices.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?" # TODO: I think this can happen, figure out why later
                assert e - num_to_crop >= 0, "this should never happen right?" # TODO: I think this can happen, figure out why later
            else:
                new_tokens.append(t)  # no truncation needed
                new_start_indices.append(s)
                new_end_indices.append(e)
        tokens, start_indices, end_indices = new_tokens, new_start_indices, new_end_indices

    # stack up exmaples into a batch
    pad_token_id = tokenizer.get_bos_token_id() # using bos as pad is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # forward the model, get the autoregressive loss and argmax pred at each token
    losses, predictions = forward_model(model, input_ids)

    # check losses and preds are correct
    if task_type == 'language_modeling':
        # for LM, currently the batch size is 1
        si = start_indices[0]
        ei = end_indices[0]
        
        # predictions[i] is the prediction for input_ids[i+1] since we roleld it in the forward_model function
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # for MC / schema: find the option with lowest average loss
        mean_losses = [losses[i: si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_indices, end_indices))]

        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return is_correct

def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    # Uses a stride pattern to split work across processes (if world_size is 4) then:
    # Process 0: indices 0, 4, 8, 12, ...
    # Process 1: indices 1, 5, 9, 13, ...
    # Process 2: indices 2, 6, 10, 14, ...
    # Process 3: indices 3, 7, 11, 15, ...
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    
    # sync results across all processes if running distributed
    # dist.barrier(): Waits for all processes to finish.
    # dist.all_reduce(..., op=dist.ReduceOp.SUM): Sums correct across all processes. After this, each process has the combined results.
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    return mean_correct
    
    # Process 0 (rank=0): evaluates examples [0, 4, 8]
    # Process 1 (rank=1): evaluates examples [1, 5, 9]
    # Process 2 (rank=2): evaluates examples [2, 6, 10]
    # Process 3 (rank=3): evaluates examples [3, 7, 11]

    # After evaluation:
    # correct = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]  (on each process)

    # After all_reduce (SUM):
    # correct = [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]  (same, since each process only wrote to its own indices)

    # mean_correct = 8/12 = 0.667 (66.7% accuracy)