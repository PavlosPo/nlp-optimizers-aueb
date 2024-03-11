# Create a pytorch lightning class that uses the following model:
# Bert from hugging face

import torch
from torch import nn, optim
import torchmetrics
from ScoresModule import MyAccuracy
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer


class LLMModule(pl.LightningModule):

    def __init__(self, num_classes, model_name="google-bert/bert-base-cased"):
        super().__init__()
        self.LLM = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.my_accuracy = MyAccuracy()
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, input_ids, attention_mask):
        return self.LLM(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.my_accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def _common_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        scores = self.forward(input_ids, attention_mask)[0]
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        scores = self.forward(input_ids, attention_mask)[0]
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)