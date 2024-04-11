from pytorch_lightning import  Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from DataModule import DataModule
from Transformer import Transformer



if __name__ == "__main__":
    seed_everything(42)

    logger = TensorBoardLogger("logs", name="my_model")

    dm = DataModule(model_name_or_path="albert-base-v2", 
                    task_name="cola", 
                    max_seq_length=128, 
                    train_batch_size=32, 
                    eval_batch_size=32, 
                    num_workers=4)
    dm.setup("fit")
    model = Transformer(
                    model_name_or_path="albert-base-v2",
                    num_labels=dm.num_labels,
                    eval_splits=dm.eval_splits,
                    task_name=dm.task_name)

    trainer = Trainer(
                    max_epochs=2,
                    accelerator="auto",
                    devices=1,  # limiting got iPython runs
                    logger=logger,
                    strategy="fsdp")
    trainer.fit(model, datamodule=dm)
    
    # We also have validate
    # trainer.validate(model, dm)
    
    # And test
    # trainer.test(model, dm)