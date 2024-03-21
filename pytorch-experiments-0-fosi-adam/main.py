import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW
from datasets import load_dataset
from fosi import fosi_adam

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained DistilBERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Define a function to preprocess the dataset
def prepare_dataset(example):
    return tokenizer(example['sentence'], truncation=True, padding=True, return_tensors='pt')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

# Define a sample dataset (replace this with your custom dataset)
# Example: IMDB movie review dataset
dataset = load_dataset('glue', 'cola').map(prepare_dataset, batched=True)

# Split dataset into train and test sets
train_dataset = dataset['train']

test_dataset = dataset['test']
# Define data loaders
batch_size = 8
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define optimizer
base_optimizer = AdamW(model.parameters(), lr=5e-5)
optimizer = fosi_adam(base_optimizer, loss_fn, next(iter(trainloader)), num_iters_to_approx_eigs=500, alpha=0.01)
# model, params = functorch.make_functional_with_buffers(model=model)
# opt_state = optimizer.init(model.parameters())

# Define training loop
for epoch in range(3):  # Adjust number of epochs as needed
    model.train()
    for i, batch in enumerate(trainloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        #This is taking care automatically
        # optimizer.zero_grad() This is taking care automatically

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')

    # Evaluate on test set
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in testloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            print("Labels: \n", labels)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test accuracy: {accuracy:.4f}')

print('Finished Training')
