## Testing which version of is being used
import strong_reject
print(strong_reject.__file__)


## This will check if the original vicuna checkpoint ships with a chat template
from transformers import AutoTokenizer
print(AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5").chat_template)  # None means you need your own


from strong_reject.local_models import load_model
## This confirms that the tokenizer is able to apply a chat template, if one is provided
pipe = load_model("lmsys/vicuna-7b-v1.5")
print(pipe.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True))
