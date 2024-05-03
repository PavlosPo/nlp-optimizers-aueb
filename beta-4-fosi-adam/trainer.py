import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple
import torchopt
from fosi import fosi_adam_torch
import copy
from logger import CustomLogger
from icecream import ic
import pickle
import os

class CustomTrainer:
    def __init__(self, 
                original_model: torch.nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                test_loader: DataLoader,
                criterion, 
                device: torch.device,
                base_optimizer = torchopt.adam,
                base_optimizer_lr: float = 0.0001,
                num_of_fosi_optimizer_iterations: int = 150,
                epochs: int = 1,
                num_classes: int = 2,
                approx_k = 20,
                eval_steps: int = 10,
                logging_steps: int = 2):
        self.original_model = original_model
        self.base_optimizer_lr = base_optimizer_lr
        self.base_optimizer = base_optimizer(lr=self.base_optimizer_lr)
        self.num_of_fosi_optimizer_iterations = num_of_fosi_optimizer_iterations
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.params = None
        self.buffers = None
        self.optimizer = fosi_adam_torch
        self.num_classes = num_classes
        self.device = device
        self.approx_k = approx_k
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.logger = CustomLogger()

        # VALIDATIONS LOSSES in order to checkpoint the model
        self.validation_metrics = {
            'loss': None,
            'model_params': None,
            'model_buffers': None,
            'f1': 'Not Implemented Yet' # TODO: Implement F1 Score
        }
        
    def train_val_test(self):
        self.original_model.to(self.device)
        self.original_model.train()
        data = next(iter(self.train_loader))
        self.optimizer = self.optimizer(self.base_optimizer, self.loss_fn, data, 
                                        approx_k=self.approx_k , 
                                        num_iters_to_approx_eigs=self.num_of_fosi_optimizer_iterations)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        self.params = tuple(param.to(self.device) for param in self.params)
        self.opt_state = self.optimizer.init(self.params)
        # Train starts here
        global_step = 0
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:
                global_step += 1
                self.original_model.train()
                self.params, self.opt_state, loss, logits = self.step(self.params, self.buffers, batch, self.opt_state)
                # Logging
                if self.logging_steps % global_step == 0:
                    self.logger.custom_log(global_step=global_step, loss=loss, outputs=logits, labels=batch['labels'], mode='train')  # per step
                if global_step % self.eval_steps == 0: # Per 100 steps
                    total_val_loss = self.evaluate(global_step=global_step, val_loader=self.val_loader)
                    # TODO: Checkpoint the model every logging_step range of batches
                    self.checkpoint_model(total_val_loss)
                    print(f"\nTotal Validation loss: {total_val_loss}\n")

                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        test_loss = self.test(self.test_loader)
        self.logger.close()
        print(f"Test Loss: {test_loss}")

    def checkpoint_model(self, val_loss: float):
        if self.validation_metrics['loss'] is None or val_loss < self.validation_metrics['loss']:
            self.validation_metrics['loss'] = val_loss
            self.validation_metrics['model_params'] = self.params
            # Save the model checkpoint locally
            self.make_checkpoint(f"./model_checkpoint")

    def make_checkpoint(self, filepath):
        # Serialize model parameters and buffers
        
        checkpoint_dir = os.path.dirname(filepath)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        state_dict = {
            'model_params': self.params,
            'model_buffers': self.buffers,
            'optimizer_state': self.opt_state,
            'loss': self.validation_metrics['loss'],
        }

        # Save the state dictionary to a file
        # with open(filepath.strip(), 'wb') as f:
        #     pickle.dump(state_dict, f)
        torch.save(state_dict, filepath)

    @staticmethod
    def load_checkpoint(filepath):
        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at '{filepath}'")

        # Load the state dictionary from the file
        # with open(filepath.strip(), 'rb') as f:
        #     state_dict = pickle.load(f)
        state_dict = torch.load(filepath)

        # Restore the model parameters and buffers
        # Assuming model is initialized before calling load_checkpoint
        loss = state_dict['loss']
        params = state_dict['model_params']
        buffers = state_dict['model_buffers']

        return params, buffers, loss


    def fine_tune(self, trial, optuna) -> float:
        """Returns the total validation loss after training the model, in order to be used by the optimizer to fine tune.

        Args:
            trial (optuna.Trial): Optuna Trial object to be used for pruning.
            optuna (optuna): Optuna library to be used for pruning.
        Returns:
            float: total validation loss from the last model, not the best one until that epoch.
        """
        ic.enable()
        self.original_model.to(self.device)
        self.original_model.train()
        data = next(iter(self.train_loader))
        self.optimizer = self.optimizer(self.base_optimizer, self.loss_fn, data, 
                                        approx_k=self.approx_k , 
                                        num_iters_to_approx_eigs=self.num_of_fosi_optimizer_iterations)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        self.params = tuple(param.to(self.device) for param in self.params)
        self.opt_state = self.optimizer.init(self.params)
        # Train starts here
        global_step = 0
        for epoch in range(self.epochs): # This picks just the model weights of the last epoch, not the best one till that epoch.
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:
                global_step += 1
                self.original_model.train()
                self.params, self.opt_state, loss, logits = self.step(self.params, self.buffers, batch, self.opt_state)
                if global_step % self.logging_steps == 0:
                    self.logger.custom_log(global_step=global_step, loss=loss, outputs=logits, labels=batch['labels'], mode='train')
                if global_step % self.eval_steps == 0:
                    current_val_loss = self.evaluate(global_step=global_step, val_loader=self.val_loader)
                    
                    # Pruning for optuna
                    trial.report(current_val_loss, global_step)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
                    # Checkpoint the model
                    self.checkpoint_model(current_val_loss)
                    print(f"\nTotal Validation loss: {current_val_loss}\n")
                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        # total_val_loss = self.evaluate(global_step=global_step, val_loader=self.val_loader)
        self.logger.close()
        best_params, best_buffers, best_loss = self.load_checkpoint(f"./model_checkpoint") # Load best model
        print(f"Total Best Val Loss: {best_loss}")
        ic.disable()
        return best_loss

    def loss_fn(self, params, batch) -> Tuple[Tensor]:
        """Loss function that is needed for the initialization of the optimizer.
        Follows the guidelines of FOSI Implementation.
        See here : https://github.com/hsivan/fosi/blob/main/examples/fosi_torch_resnet_cifar100.py#L261"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        # TODO: Could I just use the saved functional model instead of loading from the start the make_functional_with_buffers method?
        # apply_fn, old_params, old_buffers = self.make_functional_with_buffers(self.original_model, disable_autograd_tracking=False)
        logits = self.functional_model(new_params_values=params, new_buffers_values=self.buffers, input_ids=input_ids, attention_mask=attention_mask)
        ic(logits)
        loss = torch.nn.CrossEntropyLoss()(logits, labels).to(self.device)
        if loss is None:
            print("Loss is None, but we are trying again...")
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            if loss is None:
                print('Loss is still None')
                raise ValueError("Loss is None")
        return loss
    

    def step(self, params, buffers, batch, opt_state):
        self.original_model.train() # just to be sure
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        loss, logits = self._loss_fn_with_logits(params, buffers, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        grads = torch.autograd.grad(loss, params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = torchopt.apply_updates(params, updates)
        return params, opt_state, loss, logits
    
    def _loss_fn_with_logits(self, params, buffers, input_ids, attention_mask, labels):
        """Custom loss function in order to return logits too."""
        # TODO: Could I just use the saved functional model instead of loading from the start the make_functional_with_buffers method?
        # apply_fn, old_params, old_buffers = self.make_functional_with_buffers(self.original_model, disable_autograd_tracking=False)
        
        logits = self.functional_model(new_params_values=params, new_buffers_values=buffers, input_ids=input_ids, attention_mask=attention_mask).to(self.device)
        ic(logits)
        loss = torch.nn.CrossEntropyLoss()(logits, labels).to(self.device)
        if loss is None:
            print("Loss is None, but we are trying again...")
            loss = torch.nn.CrossEntropyLoss()(logits, labels).to(self.device)
            if loss is None:
                print('Loss is still None')
                raise ValueError("Loss is None")
        return loss, logits

    
    def evaluate(self, global_step: int, val_loader: DataLoader = None):
        assert val_loader is not None, "Validation loader is required for evaluation"
        progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        self.original_model.eval()  # Set the model to evaluation mode
        total_loss = 0
        outputs_all = []
        labels_all = []
        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self._loss_fn_with_logits(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])    
                total_loss += loss.clone().detach().cpu().numpy().item()
                outputs_all.extend(logits.clone().detach().cpu().numpy())
                labels_all.extend(batch['labels'].clone().detach().cpu().numpy())
            progress_bar.set_description(f"Validation at Global Step: {global_step}, Validation Loss: {loss.item():.4f}")
        self.logger.custom_log(global_step=global_step, loss=total_loss/len(val_loader), outputs=outputs_all, labels=labels_all, mode='validation')
        return total_loss / len(val_loader)

    def test(self, test_loader: DataLoader = None):
        assert test_loader is not None, "Test loader is required for testing"
        progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader))
        self.original_model.eval()  # Set the model to evaluation mode
        total_loss = 0
        outputs_all = []
        labels_all = []
        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self._loss_fn_with_logits(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                total_loss += loss.clone().detach().cpu().numpy().item()
                outputs_all.extend(logits.clone().detach().cpu().numpy())
                labels_all.extend(batch['labels'].clone().detach().cpu().numpy())
            progress_bar.set_description(f"Test Loss: {loss.item():.4f}")
        self.logger.custom_log(global_step=1, loss=total_loss/len(test_loader), outputs=outputs_all, labels=labels_all, mode='test')
        return total_loss/len(test_loader)
    
    # def give_additional_data_for_logging(self, 
    #                                     dataset_name: str = None,
    #                                     dataset_task: str = None,
    #                                     dataset_size: int = None,
    #                                     test_dataset_size: int = None,
    #                                     validation_dataset_size: int = None,
    #                                     train_dataset_size: int = None,
    #                                     num_classes: int = None,
    #                                     optimizer_name: str = None,
    #                                     base_optimizer_name: str = None,
    #                                     learning_rate_of_base_optimizer: float = None,
    #                                     batch_size: int = None,
    #                                     epochs: int = None,
    #                                     k_approx: int = None,
    #                                     num_of_optimizer_iterations: int = None,
    #                                     range_to_select: int = None,
    #                                     seed_num: int = None) -> None: 
    #     """Creates config file for the experiment and logs it."""

    #     base_optimizer_name = type(self.base_optimizer).__name__
    #     learning_rate_of_base_optimizer = self.base_optimizer_lr
    #     optimizer_name = self.optimizer.__name__
    #     num_classes = self.num_classes
    #     if range_to_select is None:
    #         range_to_select = 'All Dataset'
    #     # Create a dictionary using locals() = local variables
    #     self.additional_information = {key: value for key, value in locals().items() if key != 'self' and value is not None}
    #     ic(self.additional_information)

    def give_additional_data_for_logging(self, **kwargs) -> None:
        """Creates config file for the experiment and logs it."""
        self.logger.log_dataset_info(**kwargs)

    def make_functional_with_buffers(self, mod, disable_autograd_tracking=False):
        """
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
        
        # Inner function
        def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
            new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
            new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
            return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args=args, kwargs=kwargs)

        if disable_autograd_tracking:
            params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)

        return fmodel, params_values, buffers_values
    
