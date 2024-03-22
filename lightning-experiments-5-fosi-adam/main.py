import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import torchopt
from fosi import fosi_adam_torch

class DistilBERTClassifier(pl.LightningModule):
    def __init__(self, num_labels=2, learning_rate=1e-3):
        super().__init__()
        
        # It is needed baed on : https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization=False   

        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self._data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        self.learning_rate = learning_rate

        # We need those in order to everything work
        self._buffers = None # Will be initialized in the training loop
        self._params = None # Will be change during each training batch
        
        self.prepare_data()

        # We need to initialize the buffers and params
        self.base_optimizer = torchopt.adam(lr=self.learning_rate)
        self.optimizer = fosi_adam_torch(self.base_optimizer, self.loss_fn, next(iter(self.first_batch_of_trainloader)), num_iters_to_approx_eigs=500, alpha=0.01)
        self.model, self._params, self._buffers = torchopt.make_functional_with_buffers(self.model)
        self.optimizer.state = self.optimizer.init(self._params)

    def loss_fn(self, params, buffers, input_ids, attention_mask, labels):
      logits = model(params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask).logits
      loss = nn.CrossEntropyLoss()(logits, labels)
      return loss
    
    def accuracy(self, params, buffers, input_ids, attention_mask, labels):
      preds = self.model(self._params, buffers=self._buffers, input_ids=input_ids, attention_mask=attention_mask).logits
      predicted_class = torch.argmax(preds, dim=1)
      return torch.sum(predicted_class == labels)


    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def prepare_data(self):
        dataset = load_dataset('glue', 'cola')

        def prepare_dataset(example):
            return self.tokenizer(example['sentence'], truncation=True, padding=True, return_tensors='pt')

        dataset.map(prepare_dataset, batched=True)

        self.train_dataset = dataset['train'].remove_columns(['sentence', 'idx']).rename_column('label', 'labels')
        self.test_dataset = dataset['test'].remove_columns(['sentence', 'idx']).rename_column('label', 'labels')
        self.first_batch_of_trainloader = next(iter(torch.utils.data.DataLoader(self.train_dataset, batch_size=64, shuffle=True, collate_fn=self._data_collator)))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=64, 
                                           shuffle=True,
                                           collate_fn=self._data_collator)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=64,
                                           collate_fn=self._data_collator)

    # def configure_optimizers(self):
    #     # Return empty list to deactivate optimization
    #     return []
        

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        loss = self.loss_fn(self._params, self._buffers, input_ids, attention_mask, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Manually update parameters
        print("Calculating Gradients\n")
        grads = torch.autograd.grad(loss, self._params)
        print("Calculating updates in the model...\n")
        updates, self.optimizer.state = self.optimizer.update(grads, self.optimizer.state, self._params)
        
        print("Finding the updates of the model finished...\n")
        print("Applying updating\n")
        self._params = torchopt.apply_updates(self._params, updates, inplace=True)
        
        return loss

    # def validation_step(self, batch, batch_idx):
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['label']
        
    #     logits = self(input_ids, attention_mask).logits
    #     loss = self.loss_fn(logits, labels)
    #     self.log('val_loss', loss)
    #     return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack(outputs).mean()
    #     self.log('avg_val_loss', avg_loss)

# Train the model using PyTorch Lightning Trainer
if __name__ == "__main__":
    model = DistilBERTClassifier()
    trainer = pl.Trainer()
    trainer.fit(model)
