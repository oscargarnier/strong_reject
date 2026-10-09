from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain gradient descent in two sentences."}]
out = pipe(messages, max_new_tokens=100)
print(out[0]["generated_text"][-1]["content"])