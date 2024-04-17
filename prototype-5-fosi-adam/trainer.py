import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple
import torchopt
from fosi import fosi_adam_torch
import copy
from logger import CustomLogger

class CustomTrainer:
    def __init__(self, original_model: torch.nn.Module, 
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                criterion, device: torch.device,
                base_optimizer = torchopt.adam(lr=0.00001),
                epochs: int = 1,
                num_classes: int = 2):
        self.original_model = original_model
        self.base_optimizer = base_optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.params = None
        self.buffers = None
        self.optimizer = None
        self.num_classes = num_classes
        self.device = device
        self.logger = CustomLogger(len_train_loader=len(self.train_loader), 
                                   len_validation_loader=len(self.val_loader), 
                                   len_test_loader=len(self.test_loader))


    def train_val_test(self):
        self.original_model.to(self.device)
        self.original_model.train()
        # torch.set_grad_enabled(True)

        # Get a batch of data to initialize the optimizer
        # This is required to initialize the FOSI optimizer 
        data = next(iter(self.train_loader))
        self.optimizer = fosi_adam_torch(self.base_optimizer, self.loss_fn, data, approx_k=20 ,num_iters_to_approx_eigs=500, alpha=0.01)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        self.params = tuple(param.to(self.device) for param in self.params)
        self.opt_state = self.optimizer.init(self.params)
        val_loss_in_this_epoch = 0
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.original_model.train()
                self.params, self.opt_state, loss, logits = self.step(self.params, self.buffers, batch, self.opt_state)

                # Log metrics for the current batch
                self.logger.custom_log(epoch=epoch, batch_idx=i, loss=loss, outputs=logits, labels=batch['labels'])

                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
            val_loss_in_this_epoch = self.evaluate(self.val_loader)
            print(f"Epoch: {epoch+1}, Validation Loss: {val_loss_in_this_epoch}")
        test_loss = self.test(self.test_loader)
        self.logger.close()
        print(f"Test Loss: {test_loss}")

    def loss_fn(self, params: Tuple[Tensor], buffers: Tuple[Tensor], input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        apply_fn, params, buffers = self.make_functional_with_buffers(self.original_model, new_params_values=params, new_buffers_values=buffers, disable_autograd_tracking=False)
        logits = apply_fn(input_ids=input_ids, attention_mask=attention_mask).to(self.device)
        loss = torch.nn.CrossEntropyLoss()(logits.squeeze(), labels.squeeze()).to(self.device)
        return loss, logits
    

    def step(self, params, buffers, batch, opt_state):
        self.original_model.train()
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        # Calculate loss
        loss, logits = self.loss_fn(params, buffers, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        grads = torch.autograd.grad(loss, params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = torchopt.apply_updates(params, updates, inplace=True)
        return params, opt_state, loss, logits

    
    def evaluate(self, val_loader: DataLoader = None):
        assert val_loader is not None, "Validation loader is required for evaluation"
        progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        self.original_model.eval()
        total_loss = 0
        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self.loss_fn(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])    
                total_loss += loss.item()
                self.logger.custom_log_validation(epoch=0, batch_idx=i, loss=loss, outputs=logits, labels=batch['labels'])
            progress_bar.set_description(f"Validation Epoch: {i+1}, Validation Loss: {loss.item():.4f}")
        return torch.mean(torch.tensor(total_loss).to(self.device)/len(val_loader))
            
    def test(self, test_loader: DataLoader = None):
        assert test_loader is not None, "Test loader is required for testing"
        progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader))
        self.original_model.eval()
        total_loss = 0
        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self.loss_fn(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])    
                self.logger.custom_log_test(batch_idx=i, loss=loss, outputs=logits, labels=batch['labels'])
            total_loss += loss.item()
            progress_bar.set_description(f"Test Epoch: {i+1}, Test Loss: {loss.item():.4f}")
            # print(f"Test Loss: {total_loss/len(test_loader)}")
        return torch.mean(torch.tensor(total_loss).to(self.device)/len(test_loader))



    
    # def _get_preds(self, input_ids: Tensor, attention_mask: Tensor, new_params: Tuple[Tensor], new_buffers: Tuple[Tensor]) -> Tensor:
    #     with torch.no_grad():
    #         apply_fn, params, buffers = self.make_functional_with_buffers(self.original_model, new_params_values=new_params, new_buffers_values=new_buffers, disable_autograd_tracking=False)
    #         y_probs = apply_fn(input_ids=input_ids, attention_mask=attention_mask)
    #         y_preds = torch.argmax(y_probs.squeeze(), dim=1).to(torch.float32).to(device=self.device)
    #         return  y_preds, y_probs

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

        return fmodel, params_values, buffers_values
    
    # def check_params_without_grad_fn(self, params):
    #     params_without_grad_fn = []
    #     for param in params:
    #         if param.grad_fn is None:
    #             params_without_grad_fn.append(param)
    #     return params_without_grad_fn

    # def check_params_with_grad_fn(self, params):
    #     params_with_grad_fn = []
    #     for param in params:
    #         if param.grad_fn is not None:
    #             params_with_grad_fn.append(param)
    #     return params_with_grad_fn
    
