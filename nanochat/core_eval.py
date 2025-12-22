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
