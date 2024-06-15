import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple
import torchopt
from fosi import fosi_sgd
import copy
from logger import CustomLogger
from icecream import ic
import numpy as np
import evaluate 
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
                # base_optimizer,
                base_optimizer_lr: float = 0.0001,
                # num_of_fosi_optimizer_iterations: int = 150,
                epochs: int = 1,
                num_classes: int = 2,
                # approx_k = 20,
                eval_steps: int = 10,
                logging_steps: int = 2):
        self.original_model = original_model
        self.base_optimizer_lr = base_optimizer_lr
        self.optimizer = torchopt.sgd(lr=self.base_optimizer_lr)
        self.opt_state = self.optimizer.init(self.original_model.parameters())
        # self.base_optimizer = base_optimizer(lr=self.base_optimizer_lr)
        # self.num_of_fosi_optimizer_iterations = num_of_fosi_optimizer_iterations
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.params = None
        self.buffers = None
        # self.optimizer = fosi_sgd
        self.num_classes = num_classes
        self.device = device
        # self.approx_k = approx_k
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.logger = CustomLogger()

        # VALIDATIONS LOSSES in order to checkpoint the model
        self.saved_metrics_as_best = {
            'loss': None,
            'model_params': None,
            'model_buffers': None,
            'f1': None
        }
        
    def train_val_test(self):
        self.original_model.to(self.device)
        self.original_model.train()
        data = next(iter(self.train_loader))
        # self.optimizer = self.optimizer(self.base_optimizer, self.loss_fn, data, 
        #                                 approx_k=self.approx_k , 
        #                                 num_iters_to_approx_eigs=self.num_of_fosi_optimizer_iterations)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        # self.params = tuple(param.to(self.device) for param in self.params)
        # self.buffers = tuple(buffer.to(self.device) for buffer in self.buffers)
        # self.opt_state = self.optimizer.init(self.params)
        # Train starts here
        self.global_step = 0
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:
                self.global_step += 1
                self.original_model.train()
                self.params, loss, logits = self.step(self.params, self.buffers, batch)
                # Logging
                if ( self.global_step == 1 ) or self.global_step % self.logging_steps  == 0:
                    self.logger.custom_log(global_step=self.global_step, loss=loss, outputs=logits, labels=batch['labels'], mode='train')  # per step
                if ( self.global_step == 1 ) or self.global_step % self.eval_steps == 0: # Per 100 steps
                    results = self.evaluate(val_loader=self.val_loader)
                    total_val_loss = results['LOSS']
                    self.checkpoint_model(val_loss=total_val_loss)
                    print(f"\nTotal Validation loss: {total_val_loss}\n")

                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        test_loss = self.test(self.test_loader)
        self.logger.close()
        print(f"Test Loss: {test_loss}")

    def checkpoint_model(self, val_loss = None, f1 = None):
        """Checkpoints the model based on the validation loss or F1 score.

        Args:
            val_loss (float, optional): If given will save the new model params based on lower loss. Defaults to None.
            f1 (float, optional): If f1 is given will try to save the new better f1 score model and values e.t.c. Defaults to None.
        """
        if f1 is not None:
            if self.saved_metrics_as_best['f1'] is None or f1 > self.saved_metrics_as_best['f1']: # If given F1 is better than the previous saved one
                self.saved_metrics_as_best['loss'] = val_loss  # If given, update the loss
                self.saved_metrics_as_best['f1'] = f1
                self.saved_metrics_as_best['model_params'] = self.params
                self.make_checkpoint(f"./model_checkpoint")
        else:   # If F1 is not given, only update the loss
            if self.saved_metrics_as_best['loss'] is None or val_loss < self.saved_metrics_as_best['loss']:
                self.saved_metrics_as_best['loss'] = val_loss
                self.saved_metrics_as_best['model_params'] = self.params
                self.make_checkpoint(f"./model_checkpoint")

    def make_checkpoint(self, filepath):
        # Serialize model parameters and buffers
        checkpoint_dir = os.path.dirname(filepath)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        state_dict = {
            'model_params': self.params,
            'model_buffers': self.buffers,
            'loss': self.saved_metrics_as_best['loss'],
            'f1': self.saved_metrics_as_best['f1']
        }
        print(f"Found Better model,\nSaving model checkpoint at {filepath}.\n")
        torch.save(state_dict, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at '{filepath}'")
        state_dict = torch.load(filepath)
        loss = state_dict['loss']
        params = state_dict['model_params']
        buffers = state_dict['model_buffers']
        f1 = state_dict['f1']
        return params, buffers, loss, f1
    
    def clean_checkpoint(self, filepath='./model_checkpoint'):
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed the checkpoint file at {filepath}")

    def clean_if_something_happens(self):
        self.clean_checkpoint("./model_checkpoint")
        self.logger.close()



    def fine_tune(self, trial, optuna) -> float:
        """Returns the total validation loss after training the model, in order to be used by the optimizer to fine tune.

        Args:
            trial (optuna.Trial): Optuna Trial object to be used for pruning.
            optuna (optuna): Optuna library to be used for pruning.
        Returns:
            float: total validation loss from the last model, not the best one until that epoch.
        """
        self.original_model.to(self.device)
        self.original_model.train()
        data = next(iter(self.train_loader))
        self.optimizer = self.optimizer(self.base_optimizer, self.loss_fn, data, 
                                        approx_k=self.approx_k , 
                                        num_iters_to_approx_eigs=self.num_of_fosi_optimizer_iterations, device=self.device)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        # self.params = tuple(param.to(self.device) for param in self.params)
        # self.buffers = tuple(buffer.to(self.device) for buffer in self.buffers)
        # self.opt_state = self.optimizer.init(self.params)
        self.global_step = 0
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:
                self.global_step += 1
                self.original_model.train()
                self.params, loss, logits = self.step(self.params, self.buffers, batch)
                if (self.global_step == 1) or (self.global_step % self.logging_steps == 0):
                    self.logger.custom_log(global_step=self.global_step, loss=loss, outputs=logits, labels=batch['labels'], mode='train')
                if (self.global_step % self.eval_steps == 0):
                    results = self.evaluate(val_loader=self.val_loader)
                    current_val_loss = results['LOSS']
                    # Pruning for optuna
                    trial.report(current_val_loss, self.global_step)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    # Checkpoint the model
                    self.checkpoint_model(val_loss=current_val_loss)
                    print(f"\nTotal Validation loss: {current_val_loss}\n")
                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        best_params, best_buffers, best_loss, best_f1 = self.load_checkpoint(f"./model_checkpoint") # Load best model
        test_loss = self.test(self.test_loader)
        self.clean_checkpoint("./model_checkpoint")  # Clean the checkpoint
        self.logger.close()
        print(f"Total Best Val Loss: {best_loss}")
        print(f"Total Best Test loss: {test_loss}")
        return best_loss
    
    def fine_tune_based_on_f1(self, trial, optuna) -> float:
        """Returns the total validation loss after training the model, in order to be used by the optimizer to fine tune.

        Args:
            trial (optuna.Trial): Optuna Trial object to be used for pruning.
            optuna (optuna): Optuna library to be used for pruning.
        Returns:
            float: total validation loss from the last model, not the best one until that epoch.
        """
        self.original_model.to(self.device)
        self.original_model.train()
        data = next(iter(self.train_loader))
        self.optimizer = self.optimizer(self.base_optimizer, self.loss_fn, data, 
                                        approx_k=self.approx_k , 
                                        num_iters_to_approx_eigs=self.num_of_fosi_optimizer_iterations, device=self.device)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        # self.params = tuple(param.to(self.device) for param in self.params)
        # self.buffers = tuple(buffer.to(self.device) for buffer in self.buffers)
        # self.opt_state = self.optimizer.init(self.params)
        self.global_step = 0
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader))
            for i, batch in progress_bar:
                self.global_step += 1
                self.original_model.train()
                self.params,  loss, logits = self.step(self.params, self.buffers, batch)
                if self.global_step % self.logging_steps == 0:
                    self.logger.custom_log(global_step=self.global_step, loss=loss, outputs=logits, labels=batch['labels'], mode='train')
                if self.global_step % self.eval_steps == 0:
                    results = self.evaluate(val_loader=self.val_loader)
                    current_val_loss = results['LOSS']
                    current_val_f1 = results['F1_Macro']
                    # Pruning for optuna
                    trial.report(current_val_f1, self.global_step)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    # Checkpoint the model
                    self.checkpoint_model(f1=current_val_f1, val_loss=current_val_loss)
                    print(f"\nTotal Validation F1: {current_val_f1}\n")
                    print(f"\nTotal Validation loss: {current_val_loss}\n")
                progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
        self.logger.close()
        best_params, best_buffers, best_loss, best_f1 = self.load_checkpoint(f"./model_checkpoint")
        print(f"Total Best Val F1: {best_f1}")
        print(f"Total Best Val Loss: {best_loss}")
        return best_f1

    def loss_fn(self, params, batch) -> Tuple[Tensor]:
        """Loss function that is needed for the initialization of the optimizer.
        Follows the guidelines of FOSI Implementation.
        See here : https://github.com/hsivan/fosi/blob/main/examples/fosi_torch_resnet_cifar100.py#L261"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].to(self.device)
        logits = self.functional_model(new_params_values=params, new_buffers_values=self.buffers, input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.CrossEntropyLoss()(logits, labels).to(self.device)
        if torch.isnan(loss):
            print(f"\n\n{'*'*50}\n\nLoss is NaN, retrying to calculate one more time...\n\n{'*'*50}\n\n")
            loss = torch.nn.CrossEntropyLoss()(logits, labels).to(self.device)
            if torch.isnan(loss):
                print(f"\n\n{'*'*50}\n\nLoss is still NaN, raising an error...\n\n{'*'*50}\n\n")
                #logits and labels print for debugging
                ic.enable()
                ic(logits)
                ic(labels)
                ic(loss)
                ic(type(loss))
                ic.disable()
                raise ValueError("Loss is NaN")
        return loss
    

    def step(self, params, buffers, batch):
        self.original_model.train()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'].to(self.device)
        loss, logits = self._loss_fn_with_logits(params, buffers, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Ensure params are on the device
        params = tuple(param.to(self.device) for param in params)
        
        # # Update parameters
        # self.optimizer.zero_grad()
        # loss.backward()
        # params = self.optimizer.step()
        # # params = tuple(param.to(self.device) for param in params)
        grads = torch.autograd.grad(loss, params)
        updates, self.opt_state = self.optimizer.update(updates=grads, params=params, state=self.opt_state, )
        params = torchopt.apply_updates(params, updates, inplace=True)


        return params,  loss, logits
    
    def _loss_fn_with_logits(self, params, buffers, input_ids, attention_mask, labels):
        """Custom loss function in order to return logits too."""
        # params = tuple(param.to(self.device) for param in params)
        # buffers = tuple(buffer.to(self.device) for buffer in buffers)
        # input_ids = input_ids.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        logits = self.functional_model(new_params_values=params, new_buffers_values=buffers, input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        if torch.isnan(loss):
            print(f"\n\n{'*'*50}\n\nLoss is NaN, retrying to calculate one more time...\n\n{'*'*50}\n\n")
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            if torch.isnan(loss):
                print(f"\n\n{'*'*50}\n\nLoss is still NaN, raising an error...\n\n{'*'*50}\n\n")
                #logits and labels print for debugging
                ic.enable()
                ic(logits)
                ic(labels)
                ic(loss)
                ic(type(loss))
                ic.disable()
                raise ValueError("Loss is NaN")
        return loss.to(self.device), logits.to(self.device)

    
    def evaluate(self, val_loader: DataLoader = None):
        assert val_loader is not None, "Validation loader is required for evaluation"
        progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        self.original_model.eval()
        total_loss = 0
        outputs_all = []
        labels_all = []
        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self._loss_fn_with_logits(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])    
                total_loss += loss.clone().detach().cpu().numpy().item()
                outputs_all.extend(logits.clone().detach().cpu().numpy())
                labels_all.extend(batch['labels'].clone().detach().cpu().numpy())
            progress_bar.set_description(f"Validation at Global Step: {self.global_step}, Validation Loss: {loss.item():.4f}")
        # Logging
        
        # ic(outputs_all)
        # ic(labels_all)

        self.logger.custom_log(global_step=self.global_step, loss=total_loss/len(val_loader), outputs=outputs_all, labels=labels_all, mode='validation')
        metrics = self.logger.return_metrics()
        ic(metrics)
        return metrics
    
    # def evaluate_based_on_f1(self, val_loader: DataLoader = None):
    #     assert val_loader is not None, "Validation loader is required for evaluation"
    #     progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
    #     self.original_model.eval()
    #     total_loss = 0
    #     outputs_all = []
    #     labels_all = []
    #     for i, batch in progress_bar:
    #         with torch.no_grad():
    #             loss, logits = self._loss_fn_with_logits(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
    #             total_loss += loss.clone().detach().cpu().numpy().item()
    #             outputs_all.extend(logits.clone().detach().cpu().numpy())
    #             labels_all.extend(batch['labels'].clone().detach().cpu().numpy())
    #         progress_bar.set_description(f"Validation at Global Step: {self.global_step}, Validation Loss: {loss.item():.4f}")
    #     self.logger.custom_log(global_step=self.global_step, loss=total_loss/len(val_loader), outputs=outputs_all, labels=labels_all, mode='validation')
    #     return total_loss / len(val_loader)

    def test(self, test_loader: DataLoader = None):
        assert test_loader is not None, "Test loader is required for testing"
        progress_bar = tqdm(enumerate(test_loader, 0), total=len(test_loader))
        self.original_model.eval()
        total_loss = 0
        outputs_all = []
        labels_all = []

        # Load best model first, this will load the correct params in the self
        loaded_params, loaded_buffers, loaded_loss, loaded_f1 = self.load_checkpoint(f"./model_checkpoint")

        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self._loss_fn_with_logits(loaded_params, buffers=loaded_buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                total_loss += loss.clone().detach().cpu().numpy().item()
                outputs_all.extend(logits.clone().detach().cpu().numpy())
                labels_all.extend(batch['labels'].clone().detach().cpu().numpy())
            progress_bar.set_description(f"Test Loss: {loss.item():.4f}")
        self.logger.custom_log(global_step=self.global_step, loss=total_loss/len(test_loader), outputs=outputs_all, labels=labels_all, mode='test')
        return total_loss/len(test_loader)

    def give_additional_data_for_logging(self, **kwargs) -> None:
        """Creates config file for the experiment and logs it."""
        self.logger.log_additional_information(**kwargs)

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