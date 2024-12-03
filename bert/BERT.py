from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert")
model = BertForSequenceClassification.from_pretrained("fine_tuned_bert")
model.eval()


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\d', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

class_labels = {0: "human", 1: "AI"}

results = []
f = open("inputs.txt", "r")
for sentence in f.readlines():
    sentence = preprocess_text(sentence)
    if not sentence:
        continue

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    results.append(f"{class_labels[predicted_class]}")

with open("outputs.txt", "w") as f:
    f.write("\n".join(results))
