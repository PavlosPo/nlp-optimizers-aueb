from model import BertClassifier
from dataset import CustomDataLoader
from training import CustomTrainer
from evaluation import CustomEvaluator
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
    num_classes=3 if dataset_task.startswith("mnli") else 1 if dataset_task == "stsb" else 2
)

# Prepare dataset
custom_dataloader = CustomDataLoader(
    dataset_from=dataset_from,
    model_name=model_name,
    dataset_task=dataset_task,
    seed_num=seed_num,
    range_to_select=100,  # Default value for now, you can prompt the user for this too if needed
    batch_size=8  # Default value for now, you can prompt the user for this too if needed
)
train_loader, val_loader, test_loader, metric = custom_dataloader.get_custom_data_loaders()

# Train model
trainer = CustomTrainer(original_model, 
    train_loader, 
    val_loader, 
    epochs=epochs)
functional_model, params, buffers = trainer.train()  # Get functional model, params, and buffers

trainer.test(test_loader=test_loader)
