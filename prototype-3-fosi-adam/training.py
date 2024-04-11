import torch
from tqdm import tqdm
from fosi import fosi_adam_torch
import copy
import torchopt
import functorch
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
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        # self.functional_model, self.params, self.buffers = torch.func.functional_call(self.original_model, dict(self.original_model.named_parameters()))
        
        opt_state = optimizer.init(self.params)   

        self.original_model.train()
        # self.functional_model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0  # Reset epoch loss for each epoch
            epoch_preds = []
            epoch_labels = []
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, data in progress_bar:
                self.original_model.train()
                progress_bar.set_description(f'Epoch {epoch+1}/{self.epochs}, Step {i}/{len(self.train_loader)}')

                input_ids = data['input_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)

                # Calculate loss, with params from previous iteration
                loss, _ = self.loss_fn(self.params, self.buffers, input_ids, attention_mask, labels)
                epoch_loss += loss.item()  # Accumulate loss for each batch

                # Calculate gradients based on loss value
                grads = torch.autograd.grad(loss, self.params)
                updates, opt_state = optimizer.update(grads, opt_state, self.params)
                self.params = torchopt.apply_updates(self.params, updates, inplace=True)

                # Bar responsible
                progress_bar.set_postfix(loss=loss.item())

                # Get predictions with updated params
                self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(mod=self.original_model, new_params_values=self.params, new_buffers_values=self.buffers)
                preds = self.functional_model(input_ids=input_ids, attention_mask = attention_mask)
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

    # def loss_fn(self, functional_model: callable, params: Tuple[Tensor], buffers: Tuple[Tensor], input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
    #     preds = functional_model(params=params, buffers=buffers, input_ids=input_ids, attention_mask=attention_mask)
    #     loss = torch.nn.functional.binary_cross_entropy(preds.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
    #     return loss

    def loss_fn(self, params: Tuple[Tensor], buffers: Tuple[Tensor], input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
        fmodel, _, __ = self.make_functional_with_buffers(mod=self.original_model, new_params_values=params, new_buffers_values=buffers)
        preds = fmodel(input_ids=input_ids, attention_mask = attention_mask)
        loss = torch.nn.functional.binary_cross_entropy(preds.squeeze().to(torch.float32), labels.squeeze().to(torch.float32))
        return loss, preds

    def validate(self) -> Tuple[float, list, list]:
        # Implement validation check here, this will run per epoch, it is NOT a test functionality.
        # This function should return validation loss, predictions, and labels for validation set
        # self.original_model.eval()  # Set model to evaluation mode
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

                # fmodel, _, __ = self.make_functional_with_buffers(mod=self.original_model, new_params_values=self.params, new_buffers_values=self.buffers)
                # preds = fmodel(input_ids=input_ids, attention_mask = attention_mask)
                loss, preds = self.loss_fn(self.params, self.buffers, input_ids, attention_mask, labels)
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

                # fmodel, _, __ = self.make_functional_with_buffers(mod=self.original_model, new_params_values=self.params, new_buffers_values=self.buffers)
                # preds = fmodel(input_ids=input_ids, attention_mask = attention_mask)
                loss, preds = self.loss_fn(self.params, self.buffers, input_ids, attention_mask, labels)
                test_loss += loss.item()  # Accumulate test loss

                predictions = torch.round(preds).to(torch.float32)
                test_preds.extend(predictions.detach().cpu().numpy())
                test_labels.extend(labels.detach().cpu().numpy())

        test_loss /= len(self.test_loader)  # Calculate average test loss
        # self.original_model.train() # Make train mode again for the next loop, if there is any

        # Log test metrics
        self.logger.log_test_metrics(test_loss, test_preds, test_labels)

        return self.functional_model, self.params, self.buffers

    # def make_functional(self, mod, new_params_values=None, disable_autograd_tracking=False):
    #     params_dict = dict(mod.named_parameters())
    #     params_names = params_dict.keys()
    #     params_values = tuple(params_dict.values())
        
    #     stateless_mod = copy.deepcopy(mod)
    #     stateless_mod.to('meta')

    #     # This remains Unchanged and not used in the code
    #     def fmodel(new_params_values=new_params_values, *args, **kwargs):
    #         if new_params_values is None:
    #             # This is the first call to the functional model
    #             new_params_values = params_values
    #         new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
    #         return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)
    
    #     if disable_autograd_tracking:
    #         params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    #     return fmodel, params_values

    def make_functional_with_buffers(self, mod, new_params_values=None, new_buffers_values=None, disable_autograd_tracking=False):

        """
        Given a module, return a functional version of the module that can be called with
        the parameters and buffers as arguments. This is useful for optimization libraries
        that require a functional interface to the model.

        Args:
            mod: A PyTorch module.
            disable_autograd_tracking: If True, the parameters will be detached from the computation graph.

        Returns:
            A tuple (fmodel, params, buffers), where:
            - fmodel is a functional version of the module.
            - params is a tuple of the parameters of the module.
            - buffers is a tuple of the buffers of the module.
        
        This was taken from the official PyTorch library.
        Repo Link: https://gist.github.com/zou3519/7769506acc899d83ef1464e28f22e6cf
        Original Docs: https://pytorch.org/docs/stable/func.migrating.html#function-transforms
        """
        params_dict = dict(mod.named_parameters())
        params_names = params_dict.keys()
        params_values = tuple(params_dict.values())

        buffers_dict = dict(mod.named_buffers())
        buffers_names = buffers_dict.keys()
        buffers_values = tuple(buffers_dict.values())
        
        stateless_mod = copy.deepcopy(mod)
        stateless_mod.to('meta')

        # def fmodel(new_params_values=new_buffers_values, new_buffers_values=new_buffers_values, *args, **kwargs):
        #     if new_params_values is None:
        #         # This is the first call to the functional model
        #         new_params_values = params_values
        #     if new_buffers_values is None:
        #         # This is the first call to the functional model
        #         new_buffers_values = buffers_values
        #     new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        #     new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
        #     return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)
        
        # Inner function
        def fmodel(new_params_values=new_params_values, new_buffers_values=new_buffers_values, *args, **kwargs):
            if new_params_values is None:
                # This is the first call to the functional model
                new_params_values = params_values
            if new_buffers_values is None:
                # This is the first call to the functional model
                new_buffers_values = buffers_values
            new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
            new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
            return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args=args, kwargs=kwargs)

        if disable_autograd_tracking:
            params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)

        # del stateless_mod
        return fmodel, params_values, buffers_values