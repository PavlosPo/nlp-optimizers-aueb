import torch
from model import BertClassifier
from dataset import CustomDataLoader
from trainer import CustomTrainer
from utils import set_seed

# Prompt user for dataset choice
dataset_from = "glue"

# Prompt user for model name
model_name = input("\nEnter the model name (e.g., 'distilbert-base-uncased'): ") or 'distilbert-base-uncased'

# Prompt user for dataset task
dataset_task = input("\nEnter the dataset task (e.g., 'sst2'): ") or 'sst2'

# Prompt user for seed number
seed_num = int(input("\nEnter the seed number (default is 1): ") or '1')

eval_step = int(input("\nEnter the evaluation steps (default is 10): ") or '500')

logging_steps = int(input("\nEnter the logging steps (default is 10): ") or '250')

k_approx = int(input("\nEnter the number of max eigenvalues to approximate (default is 20): ") or '20')

num_of_fosi_iterations = int(input("\nEnter the rate iteration to apply FOSI (default is every 100 steps): ") or '100')

learning_rate = float(input("\nEnter the learning rate (default is 5e-5): ") or '5e-5')

try:
    range_to_select = int(input("\nEnter the range to select (default is All Dataset): ")) 
except ValueError:
    range_to_select = None

batch_size = int(input("\nEnter the batch size (default is 4): ") or '4')

# Prompt user for number of epochs
epochs = int(input("\nEnter the number of epochs (default is 2): ") or '2')

# Set seed for reproducibility
set_seed(seed_num)

# Set device
# try: 
#     import torch_xla
#     import torch_xla.core.xla_model as xm
#     device = xm.xla_device()
#     print('XLA Will be used as device.')
# except:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 3 if dataset_task.startswith("mnli") else 1 if dataset_task=="stsb" else 2

# Load model
original_model = BertClassifier(
    model_name=model_name,
    num_labels=num_classes,
    device=device
)

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
    approx_k=k_approx,
    base_optimizer_lr=learning_rate,
    num_of_fosi_optimizer_iterations=num_of_fosi_iterations,
    eval_steps=eval_step,
    logging_steps=logging_steps)

trainer.give_additional_data_for_logging(
        dataset_name=dataset_from,
        dataset_task=dataset_task,
        num_classes=num_classes,
        dataset_size=len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        test_dataset_size=len(test_loader.dataset),
        validation_dataset_size=len(val_loader.dataset),
        train_dataset_size=len(train_loader.dataset),
        k_approx=k_approx,
        seed_num=seed_num,
        range_to_select=range_to_select,
        batch_size=batch_size,
        epochs=epochs,
        num_of_optimizer_iterations=num_of_fosi_iterations,
        learning_rate=learning_rate,
        model_name=model_name,
        device=device,
        model_type="bert",
        optimizer="fosi",
        criterion="cross_entropy",
        task_type="classification",
        mode = "training",
        eval_steps=eval_step
    )
# trainer.init_information_logger()

trainer.train_val_test()
