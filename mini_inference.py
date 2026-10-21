print("starting")
from transformers import pipeline
import torch
torch.backends.cudnn.enabled = False
print("imported")
pipe = pipeline(
    "text-generation",
    model="DavidAU/Qwen3.5-9B-Claude-4.6-HighIQ-THINKING-HERETIC-UNCENSORED",
    device_map="auto",
)
print("pipe is loaded")
messages = [{"role": "user", "content": "Explain gradient descent in two sentences."}]

out = pipe(messages, max_new_tokens=100)

print(out[0]["generated_text"][-1]["content"])
