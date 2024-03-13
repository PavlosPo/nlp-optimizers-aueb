import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import datasets
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer

from datasets import load_dataset, load_metric, concatenate_datasets

task = "cola"

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=3)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        # self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.l1 = model

    def forward(self, x):
        # return self.l1(x)
        return self.l1(x)[0]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        # self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dataset = load_dataset("glue", "cola")
metric = load_metric('glue', "cola")

dataset1 = concatenate_datasets([dataset["train"], dataset["validation"]])



# SPLIT DATA (seed = 1)
s = 1
dataset2 = dataset1.train_test_split(test_size=0.1666666666666, stratify_by_column='label')

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5, stratify_by_column='label')["train"]
test = dataset2["test"].train_test_split(test_size=0.5, stratify_by_column='label')["test"]




encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# model
model = LitAutoEncoder(Encoder(), Decoder())

# train model
# train with both splits
trainer = L.Trainer()
trainer.fit(model, encoded_train, encoded_valid)

# train_loader = DataLoader(encoded_train)
# valid_loader = DataLoader(encoded_valid)
# test_loader = DataLoader(encoded_test)


    
# Load data sets
# transform = transforms.ToTensor()
# train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
# test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)


# dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
# train_loader = DataLoader(dataset)

# use 20% of training data for validation
# train_set_size = int(len(train_set) * 0.8)
# valid_set_size = len(train_set) - train_set_size

# split the train set into two
# seed = torch.Generator().manual_seed(42)
# train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

# train_loader = DataLoader(train_set)
# valid_loader = DataLoader(valid_set)

# # model
# model = LitAutoEncoder(Encoder(), Decoder())

# # train model
# # train with both splits
# trainer = L.Trainer()
# trainer.fit(model, train_loader, valid_loader)