import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple
import torchopt
from fosi import fosi_adam_torch

class CustomTrainer:
    def __init__(self, original_model: torch.nn.Module, 
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                criterion: torch.nn.Module,
                base_optimizer: torchopt.Optimizer = torchopt.adam(lr=0.01),
                epochs: int = 1):
        self.original_model = original_model
        self.base_optimizer = base_optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        self.original_model.train()
        self.original_model.to(self.device)

        
        data = next(iter(self.train_loader))
        self.optimizer = fosi_adam_torch(self.base_optimizer, self.loss_fn, data, num_iters_to_approx_eigs=500, alpha=0.01)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        
        opt_state = self.optimizer.init(self.params) 

        for epoch in range(self.epochs):
            self.original_model.train()

            # Use tqdm for progress bar
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:

                self.params, opt_state, loss = self.step(self.params, batch, opt_state)
                
                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
            self.evaluate(self.val_loader)

    def loss_fn(self, params: Tuple[Tensor], buffers: Tuple[Tensor], input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
      # x, y = batch
      # y_pred = apply_fn(params, x)
      y_preds = self._get_preds(input_ids, attention_mask, params, buffers)
      loss = self.criterion(y_preds, labels)  # Calculate the loss
      return loss
    
    def step(self, params, batch, opt_state):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        loss = self.loss_fn(params, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        grads = torch.autograd.grad(loss, params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = torchopt.apply_updates(params, updates, inplace=True)
        return new_params, opt_state, loss
    
    def _get_preds(self, input_ids: Tensor, attention_mask: Tensor, params: Tuple[Tensor], buffers: Tuple[Tensor]) -> Tensor:
      self.original_model.to(self.device)
      self.original_model.eval()
      with torch.no_grad():
          apply_fn, params, buffers = self.make_functional_with_buffers(self.original_model, new_params_values=params, new_buffers_values=buffers)
          y_preds = apply_fn(input_ids, attention_mask)
      return y_preds

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