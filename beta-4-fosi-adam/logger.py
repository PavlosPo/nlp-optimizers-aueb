from typing import Dict
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging
from evaluate import load
import evaluate
# from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, roc_auc_score, matthews_corrcoef
import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic
import datetime

class CustomLogger:
    def __init__(self) -> None:
        self.writer = None

    def _initialize_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=f"./runs2/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def _log_metrics(self, mode, global_step, metrics):
        self._initialize_writer()
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'{metric_name}/{mode.capitalize()}', metric_value, global_step=global_step)

    def log_metrics(self, mode, global_step, **metrics):
        self._log_metrics(mode, global_step, metrics)

    def log_dataset_info(self, **dataset_info):
        self._initialize_writer()
        self.dataset_info = dataset_info
        if dataset_info:
            for key, value in dataset_info.items():
                self.writer.add_text('Dataset Information', f'{key}: {value}')

    def log_additional_information(self, additional_info):
        self._initialize_writer()
        if additional_info:
            for key, value in additional_info.items():
                self.writer.add_text('Additional Information', f'{key}: {value}')

    def close(self):
        if self.writer:
            self.writer.close()

    def custom_log(self, global_step, loss, outputs, labels, mode): # mode = 'train' , 'validation', 'test
        outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
        labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

        outputs_softmax = F.softmax(torch.tensor(outputs), dim=1)
        outputs_argmax = np.argmax(outputs_softmax, axis=1)

        self.create_and_log_values(loss, outputs_argmax, labels, global_step, mode=mode)

    def create_and_log_values(self, loss, outputs_argmax, labels, global_step, mode):
        # Calculate metrics based on evaluate function
        evaluator = load(self.dataset_info['dataset_name'].lower(), self.dataset_info['dataset_task'].lower())
        metrics = evaluator.compute(predictions=outputs_argmax, references=labels)
        # add loss to metrics
        metrics['loss'] = loss.clone().detach().cpu().numpy().item() if torch.is_tensor(loss) else loss
        metrics['f1'] = evaluate.load('f1').compute(predictions=outputs_argmax, references=labels)['f1']
        metrics['accuracy'] = evaluate.load('accuracy').compute(predictions=outputs_argmax, references=labels)['accuracy']
        metrics['precision'] = evaluate.load('precision').compute(predictions=outputs_argmax, references=labels)['precision']
        metrics['recall'] = evaluate.load('recall').compute(predictions=outputs_argmax, references=labels)['recall']
        metrics['mae'] = evaluate.load('mae').compute(predictions=outputs_argmax, references=labels)['mae']
        # metrics['auc_roc'] = evaluate.load('auc_roc').compute(predictions=outputs_argmax, references=labels)
        ic(metrics)
        self.log_metrics(mode, global_step, **metrics)