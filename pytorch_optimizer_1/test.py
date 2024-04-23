import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
from fosi_custom import FOSIOptimizer

# Step 1: Define your model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# first batch
# Generate a dummy training dataset
num_samples = 1000  # Number of samples in the dataset
seq_length = 20  # Length of each sequence
num_classes = 2  # Number of classes

# Generate random input sequences
input_sequences = torch.randint(low=0, high=1000, size=(num_samples, seq_length))

# Generate random attention masks
attention_masks = torch.randint(low=0, high=2, size=(num_samples, seq_length))

# Generate random labels
labels = torch.randint(low=0, high=num_classes, size=(num_samples,))

# Step 4: Prepare your training data and train the model
train_dataset = TensorDataset(input_sequences, attention_masks, labels)
batch_size = 32  # Your desired batch size
num_epochs = 3  # Number of epochs for training

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

batch = next(iter(train_loader))

# Step 2: Create your optimizer
custom_optimizer = FOSIOptimizer(model.parameters(), base_optimizer=optim.Adam, momentum_func=nn.functional.mse_loss, loss_fn=nn.CrossEntropyLoss(), batch=batch)

# Step 3: Define your training loop
def train(model, optimizer, train_loader, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            loss.backward()
            optimizer.step()

# Train the model
train(model, custom_optimizer, train_loader, num_epochs)