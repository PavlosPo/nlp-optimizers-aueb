import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from torch.optim import Adam
import torch.nn as nn
import torchopt
import functorch

from datasets import load_dataset
from fosi import fosi_adam_torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained DistilBERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


# Define a function to preprocess the dataset
def prepare_dataset(example):
    return tokenizer(example['sentence'], truncation=True, padding="max_length", return_tensors='pt')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Define a sample dataset (replace this with your custom dataset)
# Example: IMDB movie review dataset
dataset = load_dataset('glue', 'cola').map(prepare_dataset, batched=True)

# Split dataset into train and test sets
train_dataset = dataset['train'].select(range(16)).set_format("torch").remove_columns(['sentence', 'idx']).rename_column('label', 'labels')

test_dataset = dataset['test'].select(range(16)).set_format("torch").remove_columns(['sentence', 'idx']).rename_column('label', 'labels')


# Define data loaders
batch_size = 8
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss()

# def loss_fn(params, batch):
#     preds = model(params, batch['input_ids'], batch['attention_mask']).logits
#     loss = nn.CrossEntropyLoss()(preds, batch)
#     return loss

def loss_fn(params, buffers, input_ids, attention_mask, labels):
    logits = model(params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask).logits
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def accuracy(params, buffers, input_ids, attention_mask, labels):
    preds = model(params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask).logits
    predicted_class = torch.argmax(preds, dim=1)
    return torch.sum(predicted_class == batch[1])

# Get first batch of data
data = next(iter(train_dataset))
input_ids = data['input_ids']
attention_mask = data['attention_mask']

# Get first batch of data
data = next(iter(train_dataset))
input_ids = data['input_ids']
attention_mask = data['attention_mask']

# Train the model
model.train()
for epoch in range(1):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        if i == 0:  # Run only once to initialize optimizer
            # Define optimizer
            base_optimizer = torchopt.adam(lr=0.001)
            optimizer = fosi_adam_torch(base_optimizer, 
                                        loss_fn, 
                                        data, 
                                        num_iters_to_approx_eigs=500, 
                                        alpha=0.01)
            model, params, buffers = functorch.make_functional_with_buffers(model=model)
            opt_state = optimizer.init(params)

        print(f"\nNumber of Epoch: {i}\n")
        print(f"Data Example: {data}")
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        # optimizer.zero_grad()
        # outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        
        loss = loss_fn(params=params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask, labels=label)
        
        print(f"Calculating Gradients\n")
        grads = torch.autograd.grad(loss, params)
        # print(f"Grads: \n{grads}\n")
        print("\n")
        print("*"*100)
        print("\n")
        print(f"Calculating updates in the model...\n")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        print("Finding the updates of the model finished...\n")
        print("Applying updating\n")
        params = torchopt.apply_updates(params, updates, inplace=True)

        # print statistics
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test data: {100 * correct / total}%')
print('Finished Training')
