from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="hf:DavidAU/Qwen3.5-9B-Claude-4.6-HighIQ-THINKING-HERETIC-UNCENSORED",
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain gradient descent in two sentences."}]
out = pipe(messages, max_new_tokens=100)
print(out[0]["generated_text"][-1]["content"])