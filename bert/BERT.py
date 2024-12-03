from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert")
model = BertForSequenceClassification.from_pretrained("fine_tuned_bert")
model.eval()

# Input sentence
text = "this is an apple"

# Tokenize and convert to tensor
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Move inputs to the same device as the model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

# Define the class labels
class_labels = {0: "negative", 1: "positive"}  # Adjust based on your dataset

# Print the result
print(f"Input: {text}")
print(f"Predicted class: {class_labels[predicted_class]}")