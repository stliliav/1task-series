from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Загрузка весов LLM...")
model = AutoModelForCausalLM.from_pretrained(model_id)

print("LLM успешно скачана!")