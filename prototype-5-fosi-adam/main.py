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
seed_num = int(input("Enter the seed number (default is 42): ") or '1')

k_approx = int(input("Enter the number of max eigenvalues to approximate (default is 20): ") or '20')

try:
    range_to_select = int(input("Enter the range to select (default is All Dataset): ")) 
except ValueError:
    range_to_select = None

batch_size = int(input("Enter the batch size (default is 8): ") or '8')

# Prompt user for number of epochs
epochs = int(input("Enter the number of epochs (default is 2): ") or '2')

# Set seed for reproducibility
set_seed(seed_num)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
original_model = BertClassifier(
    model_name=model_name,
    num_labels=2,
    device=device
)

# print(f"Model: {original_model}")

# Prepare dataset
custom_dataloader = CustomDataLoader(
    dataset_from=dataset_from,
    model_name=model_name,
    dataset_task=dataset_task,
    seed_num=seed_num,
    range_to_select=range_to_select,  # Default value for now, you can prompt the user for this too if needed
    batch_size=batch_size  # Default value for now, you can prompt the user for this too if needed
)
train_loader, val_loader, test_loader = custom_dataloader.get_custom_data_loaders()

# Train model
trainer = CustomTrainer(original_model, 
    train_loader, 
    val_loader,
    test_loader,
    epochs=epochs,
    criterion=torch.nn.CrossEntropyLoss(),  # This is not be applied , it is hardcoded for now
    device=device,
    approx_k=k_approx)

trainer.give_additional_data_for_logging(
        dataset_name=dataset_from,
        dataset_task=dataset_task,
        dataset_size=len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        test_dataset_size=len(test_loader.dataset),
        validation_dataset_size=len(val_loader.dataset),
        train_dataset_size=len(train_loader.dataset),
        k_approx=k_approx,
        seed_num=seed_num,
        range_to_select=range_to_select,
        batch_size=batch_size,
        epochs=epochs,
    )
trainer.init_information_logger()



trainer.train_val_test()  # Get functional model, params, and buffers
