import pytorch_lightning as pl
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

class LLMDataModule(pl.LightningDataModule):
    def __init__(self, model_name, dataset_name, batch_size=16, num_workers=4, max_length=512):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load the dataset
        dataset = datasets.load_dataset(self.dataset_name, split="train").train_test_split(test_size=0.2)

        # Split the dataset into train, val, and test sets
        train_data, val_test_data = dataset['train'], dataset['test']
        temp_data = val_test_data.train_test_split(test_size=0.5)
        val_data, test_data = temp_data['train'], temp_data['test']

        if stage == 'fit' or stage is None:
            self.train_dataset = train_data.map(self.tokenize_data).remove_columns(["text"])
            self.val_dataset = val_data.map(self.tokenize_data).remove_columns(["text"])

        if stage == 'test' or stage is None:
            self.test_dataset = test_data.map(self.tokenize_data).remove_columns(["text"])

    def tokenize_data(self, example):
        # Tokenize the example using the tokenizer
        return self.tokenizer(example["text"], truncation=True, padding=True, max_length=self.max_length)

    
    
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     # Move the batch to the device
    #     return DataCollatorForLanguageModeling(self.tokenizer,return_tensors='pt')
    
    def pad_collate(self, batch):

        # Pad sequences to the length of the longest sequence in the batch
        padded_batch = DataCollatorForLanguageModeling(self.tokenizer,return_tensors='pt')

        return padded_batch(batch)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          collate_fn=self.pad_collate, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.pad_collate, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          collate_fn=self.pad_collate, 
                          num_workers=self.num_workers)
