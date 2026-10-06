from transformers import pipeline

classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

text = "Hey, are we still meeting for coffee at 5 pm?"
result = classifier(text)

print(f"Text: {text}")
print(f"Result: {result}")