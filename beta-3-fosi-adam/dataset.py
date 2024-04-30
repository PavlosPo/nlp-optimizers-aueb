from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset, concatenate_datasets
from typing import Tuple
from datasets import DatasetDict
from icecream import ic

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
      # self.metric = evaluate.load(self.dataset_from, self.dataset_task)

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
      # ic(self.sentence1_key, self.sentence2_key)
      # ic(self.task_to_keys[self.dataset_task])

    def get_custom_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
      # dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
      # dataset = concatenate_datasets([dataset["train"], dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=self.seed_num, stratify_by_column='label')
      # dataset['validation'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
      # dataset['test'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']

      dataset = self.get_correct_train_test_val_split_based_on_task(self.dataset_task)

      if self.range_to_select is None:  # Use the entire dataset
        train_dataset = dataset['train'].remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        val_dataset = dataset['validation'].remove_columns(['idx'] + [col for col in dataset["validation"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        test_dataset = dataset['test'].remove_columns(['idx'] + [col for col in dataset["test"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
      else: # Use a subset of the dataset
        train_dataset = dataset['train'].select(range(self.range_to_select)).remove_columns(['idx'] + [col for col in dataset["train"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        val_dataset = dataset['validation'].select(range(self.range_to_select)).remove_columns(['idx'] + [col for col in dataset["validation"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')
        test_dataset = dataset['test'].select(range(self.range_to_select)).remove_columns(['idx'] + [col for col in dataset["test"].column_names if col in self.task_to_keys[self.dataset_task]]).rename_column('label', 'labels')

      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
      val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.data_collator)

      # ic(len(train_loader), len(val_loader), len(test_loader))
      return train_loader, val_loader, test_loader
    
    def _prepare_dataset(self, examples) -> dict:
      if self.sentence2_key is None:
          return self.tokenizer(examples[self.sentence1_key], truncation=True)
      # ic(examples[self.sentence1_key][:10], examples[self.sentence2_key][:10])
      return self.tokenizer(examples[self.sentence1_key], examples[self.sentence2_key], truncation=True)
    
    def get_correct_train_test_val_split_based_on_task(self, task: str):
      """
      This function will return a DatasetDict object splited in train validation and test datasets based on the task
      """
      if task == "cola": # 0 or 1
        """
        DatasetDict({
        train: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 8551
        })
        validation: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 1043
        })
        test: Dataset({
            features: ['sentence', 'label', 'idx'],
            num_rows: 1063
          })
        })
        """
        loaded_dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
        dataset = concatenate_datasets([loaded_dataset["train"], loaded_dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=self.seed_num, stratify_by_column='label') # in the concat do not include the test dataset
        train = dataset['train']
        valid = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
        test = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
        # Prepare the dataset dictionary to return
        dataset_to_return = DatasetDict({
              'train': train,
              'validation': valid,
              'test': test
          })
        ic(dataset_to_return)
        return dataset_to_return
      elif task == "mnli": # 0, 1, 2
          """
          DatasetDict({
          train: Dataset({
              features: ['premise', 'hypothesis', 'label', 'idx'],
              num_rows: 392702
          })
          validation_matched: Dataset({
              features: ['premise', 'hypothesis', 'label', 'idx'],
              num_rows: 9815
          })
          validation_mismatched: Dataset({
              features: ['premise', 'hypothesis', 'label', 'idx'],
              num_rows: 9832
          })
          test_matched: Dataset({
              features: ['premise', 'hypothesis', 'label', 'idx'],
              num_rows: 9796
          })
          test_mismatched: Dataset({
              features: ['premise', 'hypothesis', 'label', 'idx'],
              num_rows: 9847
            })
          })

          Here are the columns in the MNLI dataset and their purposes:
          Given a premise sentence and a hypothesis sentence, 
          the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral)
          """
          loaded_dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
          a = loaded_dataset["validation_matched"].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
          b = loaded_dataset["validation_mismatched"].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
          c = loaded_dataset["validation_matched"].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
          d = loaded_dataset["validation_mismatched"].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']

          valid = concatenate_datasets([a, b])
          test = concatenate_datasets([c, d])
          train = loaded_dataset["train"].train_test_split(test_size=1 - 50000 / len(loaded_dataset["train"]), seed=self.seed_num, stratify_by_column='label')['train']
          # Prepare the dataset dictionary to return
          dataset_to_return = DatasetDict({
              'train': train,
              'validation': valid,
              'test': test
          })
          ic(dataset_to_return)
          return dataset_to_return
      elif task == "mrpc": # 0 or 1
          """
          The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
          DatasetDict({
              train: Dataset({
                  features: ['sentence1', 'sentence2', 'label', 'idx'],
                  num_rows: 3668
              })
              validation: Dataset({
                  features: ['sentence1', 'sentence2', 'label', 'idx'],
                  num_rows: 408
              })
              test: Dataset({
                  features: ['sentence1', 'sentence2', 'label', 'idx'],
                  num_rows: 1725
              })
          })
          """
          loaded_dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
          dataset = concatenate_datasets([loaded_dataset["train"], loaded_dataset["validation"], load_dataset['test']]).train_test_split(test_size=0.1666666666666, seed=self.seed_num, stratify_by_column='label')
          # dataset['validation'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
          # dataset['test'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
          # Prepare the dataset dictionary to return
          dataset_to_return = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train'],
            'test': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
          })
          ic(dataset_to_return)
          return dataset_to_return
      elif task == "qnli": # 0 or 1
          """
          DatasetDict({
              train: Dataset({
                  features: ['question', 'sentence', 'label', 'idx'],
                  num_rows: 104743
              })
              validation: Dataset({
                  features: ['question', 'sentence', 'label', 'idx'],
                  num_rows: 5463
              })
              test: Dataset({
                  features: ['question', 'sentence', 'label', 'idx'],
                  num_rows: 5463
              })
          })
          """
          loaded_dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
          dataset = concatenate_datasets([loaded_dataset["train"], loaded_dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=self.seed_num, stratify_by_column='label')
          # dataset['validation'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
          # dataset['test'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
          # Prepare the dataset dictionary to return
          dataset_to_return = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train'],
            'test': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
        })
          ic(dataset_to_return)
          return dataset_to_return
      elif task == "sst2": # 0 or 1
          """
          DatasetDict({
              train: Dataset({
                  features: ['sentence', 'label', 'idx'],
                  num_rows: 67349
              })
              validation: Dataset({
                  features: ['sentence', 'label', 'idx'],
                  num_rows: 872
              })
              test: Dataset({
                  features: ['sentence', 'label', 'idx'],
                  num_rows: 1821
              })
          })
          """
          loaded_dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
          dataset = concatenate_datasets([loaded_dataset["train"], loaded_dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=self.seed_num, stratify_by_column='label')
          # dataset['validation'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train']
          # dataset['test'] = dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
          # Prepare the dataset dictionary to return
          dataset_to_return = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['train'],
            'test': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num, stratify_by_column='label')['test']
          })
          ic(dataset_to_return)
          return dataset_to_return
      elif task == "stsb":
        """
        DatasetDict({
              train: Dataset({
                  features: ['sentence1', 'sentence2', 'label', 'idx'],
                  num_rows: 5749
              })
              validation: Dataset({
                  features: ['sentence1', 'sentence2', 'label', 'idx'],
                  num_rows: 1500
              })
              test: Dataset({
                  features: ['sentence1', 'sentence2', 'label', 'idx'],
                  num_rows: 1379
              })
          })
        """
        loaded_dataset = load_dataset(self.dataset_from, self.dataset_task).map(self._prepare_dataset, batched=True)
        dataset = concatenate_datasets([loaded_dataset["train"], loaded_dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=self.seed_num)
        # Prepare the dataset dictionary to return
        dataset_to_return = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num)['train'],
            'test': dataset['test'].train_test_split(test_size=0.5, seed=self.seed_num)['test']
        })
        ic(dataset_to_return)
        return dataset_to_return
      else:
        raise ValueError("Task not found")


       