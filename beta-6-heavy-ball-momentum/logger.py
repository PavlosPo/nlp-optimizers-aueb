from typing import Dict
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging
import wandb
from evaluate import load
import evaluate
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, roc_auc_score, matthews_corrcoef
import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic
import datetime
import copy

class CustomLogger:
    # Datetime as projectname
    def __init__(self, project_name: str = "Heavy-Ball-Experiments", experiment_name: str = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") -> None:
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run = None
        self._initialize_run()

    def _initialize_run(self):
        if self.run is None:
            wandb.init(project=self.project_name, name=self.experiment_name)
            self.run = wandb.run

    def _log_metrics(self, mode, global_step, metrics):
        wandb.log({f'{metric_name}/{mode.capitalize()}': metric_value for metric_name, metric_value in metrics.items()}, step=global_step)

    def log_metrics(self, mode, global_step, **metrics):
        self._log_metrics(mode, global_step, metrics)

    # def log_dataset_info(self, **dataset_info):
    #     self._initialize_run()
    #     if dataset_info:
    #         wandb.config.update(dataset_info)

    def log_additional_information(self, **additional_info):
        if additional_info:
            self.additional_info = additional_info
            wandb.config.update(additional_info)

    def close(self):
        wandb.finish()

    def custom_log(self, global_step, loss, outputs, labels, mode): # mode = 'train' , 'validation', 'test
        outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
        labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        # ic(outputs)
        # ic(labels)

        outputs_softmax = F.softmax(torch.tensor(outputs), dim=1)
        outputs_argmax = np.argmax(outputs_softmax, axis=1).flatten()
        ic(outputs_argmax)

        self.create_and_log_values(loss, outputs_argmax, labels, global_step, mode=mode)


    def create_and_log_values(self, loss, outputs_argmax, labels, global_step, mode):
        # Calculate metrics based on evaluate function
        self.metrics = {}
        self.metrics['LOSS'] = loss.clone().detach().cpu().numpy().item() if torch.is_tensor(loss) else loss
        self.metrics['F1_Macro'] = evaluate.load('f1').compute(predictions=outputs_argmax, references=labels)['f1']
        self.metrics['ACCURACY'] = evaluate.load('accuracy').compute(predictions=outputs_argmax, references=labels)['accuracy']
        self.metrics['PRECISION'] = evaluate.load('precision').compute(predictions=outputs_argmax, references=labels)['precision']
        self.metrics['RECALL'] = evaluate.load('recall').compute(predictions=outputs_argmax, references=labels)['recall']
        self.metrics['MAE'] = evaluate.load('mae').compute(predictions=outputs_argmax, references=labels)['mae']
        self.metrics['MCC'] = evaluate.load('matthews_correlation').compute(predictions=outputs_argmax, references=labels)['matthews_correlation']
        self.log_metrics(mode, global_step, **self.metrics)

    def return_metrics(self):
        if self.metrics:
            return copy.deepcopy(self.metrics)