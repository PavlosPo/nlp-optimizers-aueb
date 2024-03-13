
import torch.nn.functional as F
from datetime import datetime
from typing import Optional
import torch
import evaluate
from pytorch_lightning import LightningModule
from torch.optim import LBFGS
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

class Transformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = True # use manual optimization

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.linear1 = torch.nn.Linear(768, 256)
        self.linear2 = torch.nn.Linear(256, self.num_labels) ## 3 is the number of classes in this example

        # self.to_class = nn.Linear(768, self.num_labels)
        # self.metric = datasets.load_metric(
        #     "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # )
        self.metric = evaluate.load("accuracy")

    def forward(self, batch):
        # sequence_output = self.model(
        #        input_ids=batch['input_ids'], 
        #        attention_mask=batch['attention_mask'],
        #        )['logits']

        sequence_output = self.model(**batch)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

        linear2_output = self.linear2(linear1_output)

        return linear2_output
    
    # def training_step(self, batch):
    #     input_ids = batch['input_ids']
    #     target = batch['labels']
    #     output = self.model(input_ids, target)
    #     loss = F.cross_entropy(output, target.view(-1))
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss

    # def training_step(self, batch, batch_idx):
    #     optimizer = self.configure_optimizers()[0][0]
    #     optimizer.zero_grad()
    #     outputs = self(**batch)
    #     loss = outputs.loss
    #     loss.backward()
    #     optimizer.step()
    #     return loss

    def training_step(self, batch, batch_idx):
        # outputs = self.model(
        #     batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        #     labels=batch["labels"],
        # )
        outputs = self.forward(**batch)
        loss = outputs['loss']
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # outputs = self.model(
        #     batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        #     labels=batch["labels"],
        # )
        outputs = self(**batch) 
        val_loss = outputs['loss']
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

        logits = outputs['logits']
        predicted_labels = torch.argmax(logits, axis=1)
        acc = self.metric.compute(
            predictions=predicted_labels.cpu().numpy(), 
            references=batch["labels"].cpu().numpy(),)
        self.log("val_acc", acc['accuracy'], on_epoch=True, on_step=False, prog_bar=True)
        return val_loss



    # LBFGS TRY
    # def training_step(self, batch, batch_idx):
    #     def closure():
    #         self.optimizer.zero_grad()
    #         outputs = self(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         return loss

    #     optimizer = self.configure_optimizers()
    #     optimizer.step(closure)
    #     return {}


    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     outputs = self(**batch)
    #     val_loss, logits = outputs[:2]

    #     if self.hparams.num_labels > 1:
    #         preds = torch.argmax(logits, axis=1)
    #     elif self.hparams.num_labels == 1:
    #         preds = logits.squeeze()

    #     labels = batch["labels"]

    #     # Log validation loss
    #     self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)

    #     # Log other metrics if needed
    #     metrics = self.metric.compute(predictions=preds.cpu().numpy(), references=labels.cpu().numpy())
    #     for key, value in metrics.items():
    #         self.log(key, value, on_epoch=True, on_step=False, prog_bar=True)

    #     return {"loss": val_loss, "preds": preds, "labels": labels}

    # def on_validation_epoch_end(self, outputs):
    #     if self.hparams.task_name == "mnli":
    #         for i, output in enumerate(outputs):
    #             # matched or mismatched
    #             split = self.hparams.eval_splits[i].split("_")[-1]
    #             preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
    #             labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
    #             loss = torch.stack([x["loss"] for x in output]).mean()
    #             self.log(f"val_loss_{split}", loss, prog_bar=True)
    #             split_metrics = {
    #                 f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
    #             }
    #             self.log_dict(split_metrics, prog_bar=True)
    #         return loss

    #     preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
    #     labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     self.log("val_loss", loss, prog_bar=True)
    #     self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    # def configure_optimizers(self):
    #     """Prepare optimizer and schedule (linear warmup and decay)"""
    #     model = self.model
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=self.trainer.estimated_stepping_batches,
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)
    
    # LBFGS TRY
    # def configure_optimizers(self):
    #     """Prepare optimizer (LBFGS)"""
    #     model = self.model
    #     optimizer = LBFGS(model.parameters(), lr=self.hparams.learning_rate)

    #     return optimizer