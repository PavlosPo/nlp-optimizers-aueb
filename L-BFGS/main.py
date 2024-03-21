import torch
import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import LBFGS
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define your DistilBERT model with custom classifier
class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

# Load pre-trained DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Load IMDb dataset
data = datasets.load_dataset("glue", "cola")

# Extract labels from the training split
labels = data['train']['label']

max_len = 512
batch_size = 1
num_classes = 2

# Create dataset and split it into train, validation, and test sets
dataset = data.map(lambda examples: tokenizer(examples['sentence'], padding='max_length', truncation=True), batched=True)

train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']

# Create DataLoader objects for train, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = DistilBERTClassifier(num_classes=num_classes)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define LBFGS optimizer
optimizer = LBFGS(model.parameters(), history_size=7, max_iter=10, line_search_fn=True)


def compute_metrics(logits_and_labels):
  logits, labels = logits_and_labels
  predictions = np.argmax(logits, axis=-1)
  acc = np.mean(predictions == labels)
  f1 = f1_score(labels, predictions, average = 'micro')
  return {'accuracy': acc, 'f1_score': f1}

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}:")
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_dataset):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        print("Labels: ")
        print(labels)
        
        def closure():
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Compute the training loss
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(0), labels.flatten())
            running_loss += loss.item()
    
    # Print training loss for the epoch
    print(f"  Training Loss: {running_loss / len(train_dataset)}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    if len(val_dataset) > 0:  # Check if validation loader is not empty
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataset):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print validation loss and accuracy for the epoch
        print(f"  Validation Loss: {val_loss / len(val_dataset)}")
        print(f"  Validation Accuracy: {correct / total}")
    else:
        print("  Validation loader is empty.")

# Testing phase
model.eval()
test_loss = 0.0
correct = 0
total = 0
if len(test_dataset) > 0:  # Check if test loader is not empty
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataset):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print test loss and accuracy
    print(f"Test Loss: {test_loss / len(test_dataset)}")
    print(f"Test Accuracy: {correct / total}")
else:
    print("Test loader is empty.")