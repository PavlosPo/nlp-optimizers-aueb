import torch
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch.nn as nn
import torchopt
import functorch
import evaluate
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import f1_score, precision_recall_curve, auc

from fosi import fosi_adam_torch

# Load pre-trained BERT model and tokenizer
BERT_MODEL_NAME = "bert-base-uncased"

EPOCHS = 2
BATCH_SIZE = 8
DATASET_FROM = 'glue'
DATASET_TASK = 'sst2'
SEED_NUM = 42
NUM_CLASSES = 1


GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task_to_keys = {
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

sentence1_key, sentence2_key = task_to_keys[DATASET_TASK]

# Fetching pretrained model
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_CLASSES)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        probability = self.sigmoid(outputs.squeeze())
        return probability

# Instantiate the classifier
classifier = BertClassifier(bert_model, NUM_CLASSES)

# Optionally, move the model to a GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

def set_seed(seed: int):
    """
    Set random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to set.
    """
    # Set seed for random and numpy
    random.seed(seed)
    np.random.seed(seed)
    
    # Set seed for torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set seed for tensorflow (if available)
    if torch.cuda.is_available():
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass  # TensorFlow is not available

# Usage
set_seed(SEED_NUM)

# Define a function to preprocess the dataset
# def prepare_dataset(example):
#     return tokenizer(example['sentence'], add_special_tokens=True, truncation=True, padding=True, return_tensors='pt')  # noqa: F821

def prepare_dataset(examples, tokenizer=tokenizer):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

dataset = load_dataset(DATASET_FROM, DATASET_TASK).map(prepare_dataset, batched=True)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]]).train_test_split(test_size=0.1666666666666, seed=SEED_NUM, stratify_by_column='label')
metric = evaluate.load(DATASET_FROM, DATASET_TASK)

# Split dataset into train and test sets, we use the train category because the test one has labels -1 only.

train_dataset = dataset['train'].select(range(0,500)).remove_columns(['sentence', 'idx']).rename_column('label', 'labels')
test_dataset = dataset['test'].select(range(500, 1000)).remove_columns(['sentence', 'idx']).rename_column('label', 'labels')

# Define data loaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

def compute_metrics(predictions, labels, metric=metric):
    predictions = np.argmax(predictions, axis=1)
    matthews = metric.compute(predictions=predictions, references=labels)['matthews_correlation']

    f1_pos = f1_score(y_true=labels, y_pred=predictions, pos_label=1)
    f1_neg = f1_score(y_true=labels, y_pred=predictions, pos_label=0)
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')

    return {
        'matthews_correlation': matthews,
        'f1_positive': f1_pos,
        'f1_negative': f1_neg,
        'macro_f1': f1_macro,
        'micro_f1': f1_micro
    }


def loss_fn(functional_model, params, buffers, input_ids, attention_mask, labels):
    preds = functional_model(params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask)
    loss = nn.functional.binary_cross_entropy(preds.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
    return loss


# Initialize optimizer and model parameters
# Those are needed in order for fosi_adam_to run in the loop below
data = next(iter(trainloader))  # get first batch of data

print(f"input_ids: {data['input_ids']}")
print(f"attention_mask: {data['attention_mask']}")
print(f"labels: {data['labels']}")

# Define optimizer
classifier.train()
base_optimizer = torchopt.adam(lr=0.01)
optimizer = fosi_adam_torch(base_optimizer, loss_fn, data, num_iters_to_approx_eigs=500, alpha=0.01)
model, params, buffers = functorch.make_functional_with_buffers(model=classifier)
opt_state = optimizer.init(params)

# Train the model
classifier.train()
for epoch in range(EPOCHS):
    progress_bar = tqdm(enumerate(trainloader, 1), total=len(trainloader))
    for i, data in progress_bar:
        progress_bar.set_description(f'Epoch {epoch+1}/{EPOCHS}, Step {i}/{len(trainloader)}')

        input_ids = data['input_ids'].squeeze().to(device)
        attention_mask = data['attention_mask'].squeeze().to(device)
        labels = data['labels'].squeeze().to(device)
        # print(f"Labels: \n{labels}\n")

        loss = loss_fn(functional_model=model, 
                       params=params, buffers=buffers, input_ids=input_ids,attention_mask=attention_mask, labels=labels)
        # print(f"Step: {i}\n")
        # print(f"Loss: {loss}\n")

        # Calculate gradients
        grads = torch.autograd.grad(loss, params)

        # Update model parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = torchopt.apply_updates(params, updates, inplace=True)

        progress_bar.set_postfix(loss=loss.item())

# Evaluate the model
classifier.eval()
evaluation_results = []
with torch.no_grad():
    for i, data in enumerate(testloader):
        evaluation_result = {}

        input_ids = data['input_ids'].squeeze().to(device)
        attention_mask = data['attention_mask'].squeeze().to(device)
        labels = data['labels'].squeeze().to(device)

        preds = model(params, buffers, input_ids, attention_mask=attention_mask)

        predicted_labels = torch.round(preds)

        # Save the evaluation results
        evaluation_result['input_ids'] = input_ids.cpu().tolist()
        evaluation_result['attention_mask'] = attention_mask.cpu().tolist()
        evaluation_result['labels'] = labels.cpu().tolist()
        evaluation_result['preds'] = preds.cpu().tolist()
        evaluation_result['predicted_labels'] = predicted_labels.cpu().tolist()

        evaluation_results.append(evaluation_result)
        metric.add_batch(predictions=predicted_labels, references=labels)

results = metric.compute()
print(f"Results: \n{results}\n")

print('Finished Training')

print(f"\nLabels: {evaluation_results[0]['labels']}\n")
print(f"\nPreds: {evaluation_results[0]['preds']}")
print(f"\nPredicted Labels: {evaluation_results[0]['predicted_labels']}")

# Free up memory by deleting variables, tensors, and models

del tokenizer
del bert_model
del classifier
del data_collator
del dataset
del metric
del train_dataset
del test_dataset
del trainloader
del testloader
del optimizer
del model
del params
del buffers
del opt_state
del evaluation_results
del results