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

class CustomTrainer:
    def __init__(self, original_model: torch.nn.Module, 
                train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                criterion, device: torch.device,
                base_optimizer = torchopt.adam,
                base_optimizer_lr: float = 0.0001,
                num_of_fosi_optimizer_iterations: int = 500,
                epochs: int = 1,
                num_classes: int = 2,
                approx_k = 20):
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
        self.logger = CustomLogger(len_train_loader=len(self.train_loader), 
                                   len_validation_loader=len(self.val_loader), 
                                   len_test_loader=len(self.test_loader))
        self.additional_information = {}
        



    def train_val_test(self):
        self.original_model.to(self.device)
        self.original_model.train()
        # torch.set_grad_enabled(True)

        # Get a batch of data to initialize the optimizer
        # This is required to initialize the FOSI optimizer 
        data = next(iter(self.train_loader))
        self.optimizer = self.optimizer(self.base_optimizer, 
                                         self.loss_fn, 
                                         data, 
                                         approx_k=self.approx_k ,
                                         num_iters_to_approx_eigs=self.num_of_fosi_optimizer_iterations, 
                                         alpha=0.01)
        self.functional_model, self.params, self.buffers = self.make_functional_with_buffers(self.original_model)
        self.params = tuple(param.to(self.device) for param in self.params)
        self.opt_state = self.optimizer.init(self.params)
        val_loss_in_this_epoch = 0
        before_500_counter = 0
        # for epoch in range(self.epochs):
        #     progress_bar = tqdm(enumerate(self.train_loader, 0), total=len(self.train_loader))
        #     for i, batch in progress_bar:
        #         batch = {k: v.to(self.device) for k, v in batch.items()}
        #         self.original_model.train()
        #         self.params, self.opt_state, loss, logits = self.step(self.params, self.buffers, batch, self.opt_state)

        #         # Log metrics for the current batch
        #         self.logger.custom_log(epoch=epoch, batch_idx=i, loss=loss, outputs=logits, labels=batch['labels'])

        #         progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

        #     val_loss_in_this_epoch = self.evaluate(epoch, self.val_loader)
        #     print(f"Epoch: {epoch+1}, Validation Loss: {val_loss_in_this_epoch}")
        # progress_bar = tqdm(enumerate(self.train_loader, 0), total=len(self.train_loader) * self.epochs)
        
        counter = 0
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.train_loader, 1):
                if counter <= 495:
                    counter += 1
                    continue
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.original_model.train()
                self.params, self.opt_state, loss, logits = self.step(self.params, self.buffers, batch, self.opt_state)

                # Log metrics for the current batch
                self.logger.custom_log(epoch=epoch, batch_idx=i, loss=loss, outputs=logits, labels=batch['labels'])

                # progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
                print(f"Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item():.4f}")

            val_loss_in_this_epoch = self.evaluate(epoch, self.val_loader)
            print(f"Epoch: {epoch+1}, Validation Loss: {val_loss_in_this_epoch}")
        
        # Do not run the test function 
        return None
    
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
        params = torchopt.apply_updates(params, updates)
        return params, opt_state, loss, logits

    
    def evaluate(self, epoch: int, val_loader: DataLoader = None):
        assert val_loader is not None, "Validation loader is required for evaluation"
        progress_bar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        self.original_model.eval()
        total_loss = 0
        for i, batch in progress_bar:
            with torch.no_grad():
                loss, logits = self.loss_fn(self.params, buffers=self.buffers, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])    
                total_loss += loss.item()
                self.logger.custom_log_validation(epoch=epoch, batch_idx=i, loss=loss, outputs=logits, labels=batch['labels'])
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
    
    def give_additional_data_for_logging(self, 
                                        dataset_name: str = None,
                                        dataset_task: str = None,
                                        dataset_size: int = None,
                                        test_dataset_size: int = None,
                                        validation_dataset_size: int = None,
                                        train_dataset_size: int = None,
                                        num_classes: int = None,
                                        optimizer_name: str = None,
                                        base_optimizer_name: str = None,
                                        learning_rate_of_base_optimizer: float = None,
                                        batch_size: int = None,
                                        epochs: int = None,
                                        k_approx: int = None,
                                        num_of_optimizer_iterations: int = None,
                                        range_to_select: int = None,
                                        seed_num: int = None) -> None: 
        """Creates config file for the experiment

        Args:
            dataset_name (str, optional): _description_. Defaults to None.
            dataset_task (str, optional): _description_. Defaults to None.
            dataset_size (int, optional): _description_. Defaults to None.
            test_dataset_size (int, optional): _description_. Defaults to None.
            validation_dataset_size (int, optional): _description_. Defaults to None.
            train_dataset_size (int, optional): _description_. Defaults to None.
            num_labels (int, optional): _description_. Defaults to None.
            num_classes (int, optional): _description_. Defaults to None.
            optimizer_name (str, optional): _description_. Defaults to None.
            base_optimizer_name (str, optional): _description_. Defaults to None.
            learning_rate_of_base_optimizer (float, optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to None.
            epochs (int, optional): _description_. Defaults to None.
            k_approx (int, optional): _description_. Defaults to None.
            num_of_optimizer_iterations (int, optional): _description_. Defaults to None.
            range_to_select (int, optional): _description_. Defaults to None.
            seed_num (int, optional): _description_. Defaults to None.
        """

        base_optimizer_name = type(self.base_optimizer).__name__
        learning_rate_of_base_optimizer = self.base_optimizer_lr
        optimizer_name = self.optimizer.__name__
        num_of_optimizer_iterations = self.num_of_fosi_optimizer_iterations
        num_classes = self.num_classes
        if range_to_select is None:
            range_to_select = 'All Dataset'
        # Create a dictionary using locals() = local variables
        self.additional_information = {key: value for key, value in locals().items() if key != 'self' and value is not None}
        ic(self.additional_information)

    def init_information_logger(self):
        # ic(self.additional_information)
        # ic(**self.additional_information)
        if self.additional_information:
            # Dictionary is not empty
            self.logger.log_additional_information(**self.additional_information)

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
    
