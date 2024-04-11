import torch
from tqdm import tqdm
from fosi import fosi_adam_torch
import functorch
import torchopt
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple
from logger import Logger  # Import the modified Logger class for logging

class CustomTrainer:
    def __init__(self, original_model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader = None,epochs: int = 1):
        self.original_model = original_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = Logger()  # Initialize the modified Logger class for logging

    def train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.original_model.train()
        self.original_model.to(self.device)

        base_optimizer = torchopt.adam(lr=0.01)
        data = next(iter(self.train_loader))
        optimizer = fosi_adam_torch(base_optimizer, self.loss_fn, data, num_iters_to_approx_eigs=500, alpha=0.01)
        self.functional_model, self.params, self.buffers = functorch.make_functional_with_buffers(model=self.original_model)
        opt_state = optimizer.init(self.params)

        self.original_model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0  # Reset epoch loss for each epoch
            epoch_preds = []
            epoch_labels = []
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, data in progress_bar:
                progress_bar.set_description(f'Epoch {epoch+1}/{self.epochs}, Step {i}/{len(self.train_loader)}')

                input_ids = data['input_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)

                loss = self.loss_fn(self.functional_model, self.params, self.buffers, input_ids, attention_mask, labels)
                epoch_loss += loss.item()  # Accumulate loss for each batch

                for i in self.params:
                    
                    if i == len(self.params):
                        print(f"Params: {i}")
                        break
                grads = torch.autograd.grad(loss, self.params)
                for i in grads:
                    
                    if i == len(grads):
                        print(f"Grads: {i}")
                        break
                updates, opt_state = optimizer.update(grads, opt_state, self.params)
                self.params = torchopt.apply_updates(self.params, updates, inplace=True)

                progress_bar.set_postfix(loss=loss.item())

                preds = self.functional_model(params=self.params, buffers=self.buffers, input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.round(preds).to(torch.float32)
                epoch_preds.extend(predictions.detach().cpu().numpy())
                epoch_labels.extend(labels.detach().cpu().numpy())

            epoch_loss /= self.train_loader.__len__()  # Calculate average loss per epoch
            self.logger.log_train_epoch(epoch + 1, epoch_loss, epoch_preds, epoch_labels)  # Log epoch metrics using the modified Logger class

            # Perform validation check here and log validation metrics
            if self.val_loader != None:
                validation_loss, validation_preds, validation_labels = self.validate()  # Implement validate() method
                self.logger.log_validation_epoch(epoch + 1, validation_loss, validation_preds, validation_labels)

        self.logger.close()  # Close the SummaryWriter when logging is complete

        return self.functional_model, self.params, self.buffers

    def loss_fn(self, functional_model: callable, params: Tuple[Tensor], buffers: Tuple[Tensor], input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
        preds = functional_model(params=params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.binary_cross_entropy(preds.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
        return loss

    def validate(self) -> Tuple[float, list, list]:
        # Implement validation check here, this will run per epoch, it is NOT a test functionality.
        # This function should return validation loss, predictions, and labels for validation set
        self.original_model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_preds = []
        val_labels = []
        progress_bar = tqdm(enumerate(self.val_loader, 0), total=len(self.val_loader))
        with torch.no_grad():
            for i, data in progress_bar:
                progress_bar.set_postfix(val_loss=val_loss)

                input_ids = data['input_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)

                preds = self.functional_model(params=self.params, buffers=self.buffers, input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(self.functional_model, self.params, self.buffers, input_ids, attention_mask, labels)
                val_loss += loss.item()  # Accumulate validation loss

                predictions = torch.round(preds).to(torch.float32)
                val_preds.extend(predictions.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

        val_loss /= self.val_loader.__len__()  # Calculate average validation loss
        return val_loss, val_preds, val_labels

    def test(self, test_loader: DataLoader):
        # Implement test method here
        # This function should log test metrics using the modified Logger class
        self.test_loader = test_loader    
        self.original_model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_preds = []
        test_labels = []
        progress_bar = tqdm(enumerate(self.test_loader, 0), total=len(self.test_loader))
        with torch.no_grad():
            for i, data in progress_bar:
                progress_bar.set_description(f'Testing {i}/{len(self.test_loader)}')

                input_ids = data['input_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)

                preds = self.functional_model(params=self.params, buffers=self.buffers, input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(self.functional_model, self.params, self.buffers, input_ids, attention_mask, labels)
                test_loss += loss.item()  # Accumulate test loss

                predictions = torch.round(preds).to(torch.float32)
                test_preds.extend(predictions.detach().cpu().numpy())
                test_labels.extend(labels.detach().cpu().numpy())

        test_loss /= len(self.test_loader)  # Calculate average test loss
        self.original_model.train() # Make train mode again for the next loop, if there is any

        # Log test metrics
        self.logger.log_test_metrics(test_loss, test_preds, test_labels)

        return self.functional_model, self.params, self.buffers
