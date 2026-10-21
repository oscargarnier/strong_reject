from functools import lru_cache
from transformers import pipeline, TextGenerationPipeline

VICUNA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system = messages[0]['content'] %}{% set messages = messages[1:] %}"
    "{% else %}"
    "{% set system = \"A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.\" %}"
    "{% endif %}"
    "{{ bos_token + system + ' ' }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}{{ 'USER: ' + m['content'] + ' ' }}"
    "{% else %}{{ 'ASSISTANT: ' + m['content'] + eos_token }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
)

LLAMA2_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{% set system_message = false %}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
    "{% else %}"
    "{% set content = message['content'] %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

DEFAULT_TEMPLATES = {
    "vicuna": VICUNA_TEMPLATE,
    "llama2": LLAMA2_TEMPLATE,
}


@lru_cache(maxsize=None)  # load each model once per process
def load_model(
    name: str,
    device_map: str = "auto",
    dtype: str = "auto",
    chat_template: str = None,
) -> TextGenerationPipeline:
    """Load a local or Hugging Face Hub model as a text-generation pipeline.

    Args:
        name: Hub ID (e.g. "meta-llama/Llama-3.1-8B-Instruct") or a local directory.
        device_map: "auto" spreads the model over available GPUs/CPU (needs `accelerate`).
        dtype: "auto" uses the dtype stored in the checkpoint (e.g. bf16).
        chat_template: Jinja template to use if the tokenizer doesn't ship one.
    """
    pipe = pipeline(
        "text-generation",
        model=name,
        device_map=device_map,
        dtype=dtype,  # use torch_dtype= on older transformers versions
    )

    tok = pipe.tokenizer
    if tok.pad_token is None:  # avoids a warning on Llama/Vicuna-style tokenizers
        tok.pad_token = tok.eos_token

    if chat_template is not None:
        tok.chat_template = chat_template
        print(f"Using custom chat template for {name}")
    elif tok.chat_template is None:
        for key, template in DEFAULT_TEMPLATES.items():
            if key in name.lower():
                tok.chat_template = template
                break
    return pipe