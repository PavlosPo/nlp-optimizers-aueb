import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BertClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 1):
        """
        Initialize the BertClassifier.

        Args:
            model_name (str, optional): Name of the pretrained model. Defaults to "bert-base-uncased".
            num_classes (int, optional): Number of output classes. Defaults to 1.
        """
        super(BertClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load the pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)
        
        # Get the tokenizer associated with the model
        self.tokenizer = self._get_tokenizer()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the classifier.

        Args:
            input_ids (Tensor): Input token ids.
            attention_mask (Tensor): Attention mask tensor.

        Returns:
            Tensor: Probability scores for the input.
        """
        # Perform forward pass through the model
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Apply activation based on the number of classes
        if self.num_classes == 1:
            # Apply sigmoid activation
            probability = torch.sigmoid(logits.squeeze())
        else:
            # Apply softmax activation
            probability = nn.functional.softmax(logits.squeeze(), dim=1)
        return probability
    
    def _get_tokenizer(self):
        """
        Get the tokenizer associated with the model.

        Returns:
            tokenizer: The tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.model_name)
