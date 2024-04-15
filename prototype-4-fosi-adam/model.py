import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_labels: int, model_name: str = "bert-base-uncased"):
        super(BertClassifier, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
    #     logits = outputs.logits
    #     probabilities = self.softmax(logits)
    #     print(f"Probabilities in the forward method: {probabilities}")
    #     print(f"Probabilities shape in the forward method: {probabilities.shape}")
    #     return probabilities
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.to( self.device )
        # probabilities = self.softmax(logits).to(torch.float32)

        # print(f"Probabilities in the forward method: {probabilities}")
        # print(f"Probabilities shape in the forward method: {probabilities.shape}")
        return logits
    
    # def backward(self, input_ids, attention_mask, labels):
    #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #     loss = outputs.loss
    #     return loss

    # Create a function that will check the grad_fn of each tensor in this model
    def check_grad_fn(self):
        for name, param in self.named_parameters():
            print(f"Name: {name}, requires_grad: {param.requires_grad}, grad_fn: {param.grad_fn}")
