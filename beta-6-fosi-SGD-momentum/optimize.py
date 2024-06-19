import torch
import optuna
from model import BertClassifier
from dataset import CustomDataLoader
from trainer import CustomTrainer
from utils import set_seed
from icecream import ic
import time
from optuna.exceptions import TrialPruned
import gc
import os

ic.enable()

# Do not upload while hypertuning, we have problem with the syncing of optuna and wandb
os.environ["WANDB_MODE"] = "offline"


def get_user_input(prompt, default, cast_type):
    user_input = input(f"\n{prompt} (default is {default}): ")
    return cast_type(user_input) if user_input else default

def main():
    # Input user the seed 
    dataset_task = get_user_input("Enter the task to run", 'cola', str)
    seed_num = get_user_input("Enter the seed number", 1, int)
    train_epoch = get_user_input("The number of training epochs", 2, int)
    eval_step = get_user_input("The number of evaluation steps", 250, int)
    logging_steps = get_user_input("The number of logging steps", 250, int)
    batch_size = get_user_input("The batch size", 4, int)
    range_to_select = get_user_input("Enter the range to select (default is All Dataset)", None, lambda x: int(x) if x else None)

    dataset_from = "glue"
    model_name = 'distilbert-base-uncased'

    # Set device
    try: 
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print('XLA Will be used as device.')
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 3 if dataset_task.startswith("mnli") else 1 if dataset_task == "stsb" else 2

    # Set seed for reproducibility
    set_seed(seed_num)

    

    def objective(trial):
        # Define hyperparameters to tune
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2)
        # This learning rate is after hypertuned Adam in seed 1 in mrpc dataset
        k_approx = trial.suggest_int('k_approx', 1, 2)
        num_of_fosi_iterations = trial.suggest_int('num_of_fosi_iterations', 50, 1000)

        
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

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
            batch_size=batch_size
        )
        train_loader, val_loader, test_loader = custom_dataloader.get_custom_data_loaders()

        # Train model
        trainer = CustomTrainer(
            original_model, 
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
            optimizer="fosi-sgd-momentum",
            criterion="cross_entropy",
            task_type="classification",
            mode="hypertuning",
            eval_steps=eval_step,
            logging_steps=logging_steps
        )

        try:
            result = trainer.fine_tune(trial=trial, optuna=optuna)  # Return the metric you want to optimize
            del trainer
            del custom_dataloader
            del original_model
            torch.cuda.empty_cache()
            time.sleep(5)
            return result
        except Exception as e:
            trainer.clean_if_something_happens()
            del trainer
            del custom_dataloader
            del original_model
            torch.cuda.empty_cache()
            time.sleep(5)
            gc.collect()
            if isinstance(e, TrialPruned):
                print("\nTrial was pruned...\n")
                raise e  # Raise the exception to stop the trial
            else:
                print(f"\nAn exception occurred: \n{e}")
                print("\nReturning None...\n")
                return None  # Return None if another exception occurs

    if __name__ == "__main__":
        # Specify the directory where you want to store the database
        database_folder = "./databases_results/"
        os.makedirs(database_folder, exist_ok=True)  # Ensure the folder exists, create if not

        # Specify the SQLite URL with load_if_exists=True to load the existing study if it exists
        sqlite_filename = f'database_{dataset_task}_epochs_{train_epoch}_batch_{batch_size}_seed_{seed_num}.db'
        sqlite_path = os.path.join(database_folder, sqlite_filename)
        sqlite_url = f'sqlite:///{sqlite_path}'

        # Set up the median stopping rule as the pruning condition.
        study = optuna.create_study(
            study_name=f'fosi_sgd__momentum_{dataset_task}_epochs_{train_epoch}_batch_{batch_size}_seed_{seed_num}', 
            storage=sqlite_url, 
            load_if_exists=True, 
            pruner=optuna.pruners.MedianPruner()
        )

        # Optimize the study
        study.optimize(objective, n_trials=30)  # Adjust n_trials as needed

        # Save the best params to a text file
        with open(f"fosi_sgd_momentum_best_params_{dataset_task}_epochs_{train_epoch}_batch_{batch_size}_seed_{seed_num}.txt", "w") as f:
            f.write(str(study.best_params))
            f.write("\n")
            f.write(str(study.best_value))

# Run the main function
main()
