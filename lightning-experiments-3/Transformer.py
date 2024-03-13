from typing import Optional
import torch
import evaluate
from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForSequenceClassification
import torch.nn.functional as F


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
        self.automatic_optimization = True

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.LLM = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)


        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.precision_metric = evaluate.load("precision")

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.LLM(input_ids, attention_mask).logits
        # Apply softmax activation function to get class probabilities
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def training_step(self, batch, batch_idx):
        # input_ids, attention_mask, labels = batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)   # This is the forward pass
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        f1 = self.f1_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        precision = self.precision_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_acc", acc['accuracy'], on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_f1", f1['f1'], on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_precision", precision['precision'], on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        #input_ids, attention_mask, labels = batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        acc = self.accuracy_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        f1 = self.f1_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        precision = self.precision_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_acc", acc['accuracy'], on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_f1", f1['f1'], on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_precision", precision['precision'], on_epoch=True, on_step=False, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc['accuracy'], "val_f1": f1['f1'], "val_precision": precision['precision']}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.01)
