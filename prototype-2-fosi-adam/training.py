import torch
from tqdm import tqdm
from fosi import fosi_adam_torch
import functorch
import torchopt
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple

class CustomTrainer:
    def __init__(self, original_model: torch.nn.Module, train_loader: DataLoader, epochs: int = 1):
        """
        Initialize the CustomTrainer.

        Args:
            original_model (torch.nn.Module): The original model to be trained.
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            epochs (int, optional): Number of epochs for training. Defaults to 1.
        """
        self.original_model = original_model
        self.train_loader = train_loader
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Train the model using FOSI-optimizer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing functional model, parameters, and buffers.
        """
        # Set model to training mode and move to device
        self.original_model.train()
        self.original_model.to(self.device)

        # Define optimizer and get first batch of data for FOSI-optimizer initialization
        base_optimizer = torchopt.adam(lr=0.01)
        data = next(iter(self.train_loader))
        optimizer = fosi_adam_torch(base_optimizer, self.loss_fn, data, num_iters_to_approx_eigs=500, alpha=0.01)
        self.functional_model, self.params, self.buffers = functorch.make_functional_with_buffers(model=self.original_model)
        opt_state = optimizer.init(self.params)

        # Train model
        self.original_model.train()
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, data in progress_bar:
                progress_bar.set_description(f'Epoch {epoch+1}/{self.epochs}, Step {i}/{len(self.train_loader)}')

                # Move data to device
                input_ids = data['input_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)

                # Compute loss and gradients
                loss = self.loss_fn(self.functional_model, self.params, self.buffers, input_ids, attention_mask, labels)
                grads = torch.autograd.grad(loss, self.params)

                # Update model parameters
                updates, opt_state = optimizer.update(grads, opt_state, self.params)
                self.params = torchopt.apply_updates(self.params, updates, inplace=True)

                progress_bar.set_postfix(loss=loss.item())

        return self.functional_model, self.params, self.buffers

    def loss_fn(self, functional_model: callable, params: Tuple[Tensor], buffers: Tuple[Tensor], input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
        """
        Custom loss function.

        Args:
            functional_model (callable): Functional model for computing predictions.
            params (Tuple[Tensor]): Model parameters.
            buffers (Tuple[Tensor]): Model buffers.
            input_ids (Tensor): Input tensor containing token ids.
            attention_mask (Tensor): Attention mask tensor.
            labels (Tensor): Tensor containing target labels.

        Returns:
            Tensor: Computed loss tensor.
        """
        preds = functional_model(params=params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.binary_cross_entropy(preds.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
        return loss
