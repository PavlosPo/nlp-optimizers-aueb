from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset, concatenate_datasets
import evaluate
from typing import Tuple

torch.set_default_dtype(torch.float32)

class CustomDataLoader:
    def __init__(self, 
                dataset_from :str = "glue",
                dataset_task : str = "cola", 
                model_name: str = "distilbert-base-uncased",
                tokenizer: AutoTokenizer = None, 
                seed_num: int = 1, range_to_select: int | None = None, batch_size: int = 8) -> None:
      self.dataset_task = dataset_task
      self.dataset_from = dataset_from
      self.model_name = model_name
      self.seed_num = seed_num
      self.range_to_select = range_to_select
      self.batch_size = batch_size
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
      self.metric = evaluate.load(self.dataset_from, self.dataset_task)

      self.GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
      self.task_to_keys = {
          "cola": ("sentence", None),
          "mnli": ("premise", "hypothesis"),
          "mnli-mm": ("premise", "hypothesis"),
          "mrpc": ("sentence1", "sentence2"),
          "qnli": ("question", "sentence"),
          "qqp": ("question1", "question2"),
          "rte": ("sentence1", "sentence2"),
          "sst2": ("sentence", None),
          "stsb": ("sentence1", "sentence2"),
          "wnli": ("sentence1", "sentence2"),
      }
      
      self.sentence1_key, self.sentence2_key = self.task_to_keys[self.dataset_task]

    def get_custom_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, evaluate.Metric]:
      dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
      dataset = concatenate_datasets([dataset["train"], dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=self.seed_num, stratify_by_column='label')

      if self.range_to_select is None:  # Use the entire dataset
        train_dataset = dataset['train'].remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        val_dataset = dataset['train'].remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        test_dataset = dataset['train'].remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
      else: # Use a subset of the dataset
        train_dataset = dataset['train'].select(range(self.range_to_select)).remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        val_dataset = dataset['train'].select(range(self.range_to_select, 2*self.range_to_select)).remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        test_dataset = dataset['train'].select(range(2*self.range_to_select, 3*self.range_to_select)).remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')


      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
      val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
      return train_loader, val_loader, test_loader
    
    def _prepare_dataset(self, examples) -> dict:
      if self.sentence2_key is None:
          return self.tokenizer(examples[self.sentence1_key], truncation=True)
      return self.tokenizer(examples[self.sentence1_key], examples[self.sentence2_key], truncation=True)