import torch
import optuna
from model import BertClassifier
from dataset import CustomDataLoader
from trainer import CustomTrainer
from utils import set_seed
from icecream import ic

ic.disable()

set_seed(1)
ic.disable()

def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2)
    k_approx = trial.suggest_int('k_approx', 5, 20)
    num_of_fosi_iterations = trial.suggest_int('num_of_fosi_iterations', 50, 200)
    # Add more hyperparameters as needed
    dataset_from = "glue"
    dataset_task = "mrpc"
    seed_num = 1
    eval_step = 100
    model_name = 'distilbert-base-uncased'
    range_to_select = None
    batch_size = 4
    epochs = 2
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    original_model = BertClassifier(
        model_name=model_name,
        num_labels=num_classes,
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
        epochs=epochs,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        approx_k=k_approx,
        base_optimizer_lr=learning_rate,
        num_of_fosi_optimizer_iterations=num_of_fosi_iterations,
        eval_steps=eval_step)

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
            eval_steps=eval_step
        )

    return trainer.fine_tune()  # Return the metric you want to optimize

if __name__ == "__main__":
    # Specify the SQLite URL with load_if_exists=True to load the existing study if it exists
    sqlite_url = 'sqlite:///fine_tuning_database.db'
    study = optuna.create_study(study_name='fine-tuning-study', storage=sqlite_url, load_if_exists=True)

    # Optimize the study
    study.optimize(objective, n_trials=150)  # Adjust n_trials as needed

    # Save the best params to a text file
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))
