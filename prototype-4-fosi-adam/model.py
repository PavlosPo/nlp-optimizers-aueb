import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, model_name: str = "bert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = self.softmax(logits)
        print(f"Probabilities in the forward method: {probabilities}")
        print(f"Probabilities shape in the forward method: {probabilities.shape}")
        return probabilities
    
    # def backward(self, input_ids, attention_mask, labels):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     loss = outputs.loss
    #     return loss