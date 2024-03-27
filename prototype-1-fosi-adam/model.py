import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BertClassifier(nn.Module):
    def __init__(self, model_name = "bert-base-uncased", num_classes=1):
        super(BertClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = self._get_tokenizer()

    def forward(self, input_ids, attention_mask):
        self.logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        self.probability = self.sigmoid(self.logits.squeeze())
        return self.probability
    
    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
