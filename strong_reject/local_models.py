from functools import lru_cache
from transformers import pipeline, TextGenerationPipeline


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

    return pipe