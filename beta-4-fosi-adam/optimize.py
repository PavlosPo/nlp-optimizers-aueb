import torch
import optuna
from model import BertClassifier
from dataset import CustomDataLoader
from trainer import CustomTrainer
from utils import set_seed
from icecream import ic

ic.disable()

# Input user the seed 
dataset_task = str(input("Enter the task to run: (default is cola): ") or 'cola')
seed_num = int(input("\nEnter the seed number (default is 1): ") or '1')
train_epoch = int(input("The number of training epochs: (default is 2): ") or '2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_from = "glue"
eval_step = 30
model_name = 'distilbert-base-uncased'
batch_size = 4
logging_steps = 30
num_classes = 3 if dataset_task.startswith("mnli") else 1 if dataset_task == "stsb" else 2
range_to_select = 150


set_seed(seed_num)

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2)
    k_approx = trial.suggest_int('k_approx', 0, 20)
    num_of_fosi_iterations = trial.suggest_int('num_of_fosi_iterations', 0, 200)
    
    # Load model
    original_model = BertClassifier(
        model_name=model_name,
        num_labels=num_classes,
        device=device
    )

    # Prepare dataset
    custom_dataloader = CustomDataLoader(
        dataset_from=dataset_from,
        dataset_task=dataset_task,
        seed_num=seed_num,
        range_to_select=range_to_select,
        batch_size=batch_size,
    )
    train_loader, val_loader, test_loader = custom_dataloader.get_custom_data_loaders()

    # Train model
    trainer = CustomTrainer(original_model, 
        train_loader, 
        val_loader,
        test_loader,
        epochs=train_epoch,
        criterion=torch.nn.CrossEntropyLoss(),
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
            epochs=train_epoch,
            num_of_optimizer_iterations=num_of_fosi_iterations,
            learning_rate=learning_rate,
            model_name=model_name,
            device=device,
            model_type="bert",
            optimizer="fosi",
            criterion="cross_entropy",
            task_type="classification",
            eval_steps=eval_step,
            logging_steps=logging_steps
        )

    try : # Catch exceptions
        result = trainer.fine_tune(trial=trial, optuna=optuna)  # Return the metric you want to optimize
        return result
    except Exception as e: # Return None if an exception occurs
        print(f"An exception occurred: {e}")
        return None


if __name__ == "__main__":
    # Specify the SQLite URL with load_if_exists=True to load the existing study if it exists
    sqlite_url = f'sqlite:///fine_tuning_dataset_{dataset_task}_num_train_epochs_{train_epoch}.db'
    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(study_name='fine-tuning-study', 
                                storage=sqlite_url, 
                                load_if_exists=True, 
                                pruner=optuna.pruners.MedianPruner())

    # Optimize the study
    study.optimize(objective, n_trials=30)  # Adjust n_trials as needed

    # Save the best params to a text file
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))
