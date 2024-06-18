import torch
from model import BertClassifier
from dataset import CustomDataLoader
from trainer import CustomTrainer
from utils import set_seed

def get_user_input(prompt, default, cast_type):
    user_input = input(f"\n{prompt} (default is {default}): ")
    return cast_type(user_input) if user_input else default

def main():
    # Prompt user for inputs
    dataset_from = "glue"
    model_name = get_user_input("Enter the model name (e.g., 'distilbert-base-uncased')", 'distilbert-base-uncased', str)
    dataset_task = get_user_input("Enter the dataset task (e.g., 'sst2')", 'sst2', str)
    seed_num = get_user_input("Enter the seed number", 1, int)
    eval_step = get_user_input("Enter the evaluation steps", 500, int)
    logging_steps = get_user_input("Enter the logging steps", 250, int)
    # k_approx = get_user_input("Enter the number of max eigenvalues to approximate", 20, int)
    # num_of_fosi_iterations = get_user_input("Enter the rate iteration to apply FOSI", 100, int)
    learning_rate = get_user_input("Enter the learning rate", 5e-5, float)
    range_to_select = get_user_input("Enter the range to select (default is All Dataset)", None, lambda x: int(x) if x else None)
    batch_size = get_user_input("Enter the batch size", 4, int)
    epochs = get_user_input("Enter the number of epochs", 2, int)

    # Set seed for reproducibility
    set_seed(seed_num)

    # Set device
    try: 
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print('XLA Will be used as device.')
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 3 if dataset_task.startswith("mnli") else 1 if dataset_task == "stsb" else 2

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
        range_to_select=range_to_select,
        batch_size=batch_size
    )
    train_loader, val_loader, test_loader = custom_dataloader.get_custom_data_loaders()

    # Train model
    trainer = CustomTrainer(
        original_model, 
        train_loader, 
        val_loader,
        test_loader,
        epochs=epochs,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        # approx_k=k_approx,
        base_optimizer_lr=learning_rate,
        # num_of_fosi_optimizer_iterations=num_of_fosi_iterations,
        eval_steps=eval_step,
        logging_steps=logging_steps
    )

    trainer.give_additional_data_for_logging(
        dataset_name=dataset_from,
        dataset_task=dataset_task,
        num_classes=num_classes,
        dataset_size=len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        test_dataset_size=len(test_loader.dataset),
        validation_dataset_size=len(val_loader.dataset),
        train_dataset_size=len(train_loader.dataset),
        # k_approx=k_approx,
        seed_num=seed_num,
        range_to_select=range_to_select,
        batch_size=batch_size,
        epochs=epochs,
        # num_of_optimizer_iterations=num_of_fosi_iterations,
        learning_rate=learning_rate,
        model_name=model_name,
        device=device,
        model_type="bert",
        optimizer="fosi",
        criterion="cross_entropy",
        task_type="classification",
        mode="training",
        eval_steps=eval_step
    )

    trainer.train_val_test()

if __name__ == "__main__":
    main()
