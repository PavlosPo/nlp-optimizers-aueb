import torch
from LLMDataModule import LLMDataModule
from LLMModule import LLMModule
import pytorch_lightning as pl

# Make this code to run on main check name
if __name__ == "__main__":
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 784
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3

    # model = NN(input_size=input_size, num_classes=num_classes)
    model = LLMModule(num_classes=num_classes, model_name="google-bert/bert-base-cased")
    dm = LLMDataModule(model_name="google-bert/bert-base-cased", 
                       dataset_name= "imdb",
                       batch_size=batch_size)

    trainer = pl.Trainer(accelerator="gpu", devices=1, min_epochs=1, max_epochs=3, precision=16)
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)