import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from torch.optim import Adam
import torch.nn as nn
import torchopt
import functorch
import evaluate
import torch.nn.functional as F
from tqdm import tqdm

from datasets import load_dataset
from fosi import fosi_adam_torch
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import subprocess

# Load pre-trained BERT model and tokenizer
bert_model_name = "bert-base-uncased"
num_classes = 1
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

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
classifier = BertClassifier(bert_model, num_classes)

# Optionally, move the model to a GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)


# Define a function to preprocess the dataset
def prepare_dataset(example):
    return tokenizer(example['sentence'], add_special_tokens=True, truncation=True, padding=True, return_tensors='pt')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

dataset = load_dataset('glue', 'sst2').map(prepare_dataset, batched=True)
metric = evaluate.load("glue", "sst2")

# Split dataset into train and test sets, we use the train category because the test one has labels -1 only.
train_dataset = dataset['train'].select(range(0,500)).remove_columns(['sentence', 'idx']).rename_column('label', 'labels')
test_dataset = dataset['train'].select(range(500, 1000)).remove_columns(['sentence', 'idx']).rename_column('label', 'labels')

# Define data loaders
batch_size = 128
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)


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
for epoch in range(2):
    progress_bar = tqdm(enumerate(trainloader, 1), total=len(trainloader))
    for i, data in progress_bar:
        progress_bar.set_description(f'Epoch {epoch+1}/{2}, Step {i}/{len(trainloader)}')

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

print(f'Epoch: {epoch}')
results = metric.compute()
print(f"Results: \n{results}\n")

print('Finished Training')

print(f"\nLabels: {evaluation_results[0]['labels']}\n")
print(f"\nPreds: {evaluation_results[0]['preds']}")
print(f"\nPredicted Labels: {evaluation_results[0]['predicted_labels']}")

def free_up_memory():
    try:
        # Execute shell commands to sync file system and drop page caches
        subprocess.run(['sync'])
        subprocess.run(['echo', '3'], capture_output=True, text=True, check=True, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Memory freed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while freeing memory: {e.stderr}")


free_up_memory()