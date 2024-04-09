import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, model_name: str = "distilbert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.model_nam = "distilbert-base-uncased"
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification(self.model_name, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits.squeeze(), dim=1)
        return probabilities