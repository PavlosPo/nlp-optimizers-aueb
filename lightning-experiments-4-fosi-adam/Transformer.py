from typing import Optional
from fosi import fosi_adam
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
        self.LLM = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=self.num_labels)


        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.precision_metric = evaluate.load("precision")

        # Determine average parameter based on the number of labels
        if self.num_labels == 2:
            self.average = "binary"
        else:
            self.average = "macro"

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
        return self._log_everything(outputs, labels, loss, mode="train")
    
    # def on_train_end(self) -> None:
    #     self.log("train/precision", self.precision_metric.aggregate().item(), on_epoch=True, prog_bar=True)
    #     self.log("train/f1", self.f1_metric.aggregate().item(), on_epoch=True, prog_bar=True)
    #     self.log("train/accuracy", self.accuracy_metric.aggregate().item(), on_epoch=True, prog_bar=True)
        
    
    def test_step(self, batch, batch_idx):
        #input_ids, attention_mask, labels = batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        return self._log_everything(outputs, labels, loss, mode="test")
    def validation_step(self, batch, batch_idx):

        #input_ids, attention_mask, labels = batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs, labels)
        return self._log_everything(outputs, labels, loss, mode="val")

    def _log_everything(self, outputs, labels, loss, mode="test"):
        acc = self.accuracy_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        f1 = self.f1_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy(), average=self.average)
        precision = self.precision_metric.compute(predictions=torch.argmax(outputs, dim=1).cpu().numpy(), references=labels.cpu().numpy())
        self.log_dict({f'{mode}_loss': loss, f'{mode}_accuracy': acc['accuracy'], f'{mode}_f1': f1['f1'], f'{mode}_precision': precision['precision']}, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "acc": acc['accuracy'], "f1": f1['f1'], "precision": precision['precision']}

    def configure_optimizers(self, ):
        # return (self.parameters())
        return fosi_adam(torch.optim.Adam(self.parameters()), loss_fn=self.loss_fn) # How to insert batch ??
