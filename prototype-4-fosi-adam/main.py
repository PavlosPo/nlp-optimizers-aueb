import torch
from model import BertClassifier
from dataset import CustomDataLoader
from trainer import CustomTrainer
from utils import set_seed

# Prompt user for dataset choice
dataset_from = input("Enter the dataset you want to use (e.g., 'glue'): ") or 'glue'

# Prompt user for model name
model_name = input("Enter the model name (e.g., 'bert-base-uncased'): ") or 'bert-base-uncased'

# Prompt user for dataset task
dataset_task = input("Enter the dataset task (e.g., 'sst2'): ") or 'sst2'

# Prompt user for seed number
seed_num = int(input("Enter the seed number (default is 42): ") or '42')

# Prompt user for number of epochs
epochs = int(input("Enter the number of epochs (default is 2): ") or '2')

# Set seed for reproducibility
set_seed(seed_num)

# Load model
original_model = BertClassifier(
    model_name=model_name,
    num_labels=2
)

# print(f"Model: {original_model}")

# Prepare dataset
custom_dataloader = CustomDataLoader(
    dataset_from=dataset_from,
    model_name=model_name,
    dataset_task=dataset_task,
    seed_num=seed_num,
    range_to_select=200,  # Default value for now, you can prompt the user for this too if needed
    batch_size=8  # Default value for now, you can prompt the user for this too if needed
)
train_loader, val_loader, test_loader = custom_dataloader.get_custom_data_loaders()

# Train model
trainer = CustomTrainer(original_model, 
    train_loader, 
    val_loader,
    test_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    epochs=epochs)

trainer.train_val_test()  # Get functional model, params, and buffers