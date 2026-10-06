import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Tokenizer loading...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on a device: {device}")

print("Model loading...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

messages = [
    {"role": "system", "content": "You are useful and polite AI tool"},
    {"role": "user", "content": "Hi! please write a congratulation for the Christmas"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("Answer generation...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\nModel response:")
print(response)