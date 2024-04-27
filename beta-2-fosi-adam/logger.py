from typing import Dict
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, roc_auc_score, matthews_corrcoef
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
        outputs = outputs.squeeze().clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
        labels = labels.squeeze().clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

        outputs_softmax = F.softmax(torch.tensor(outputs), dim=1).numpy()
        outputs_argmax = np.argmax(outputs_softmax, axis=1)

        self.create_and_log_values(loss, outputs_argmax, labels, global_step, mode=mode)

    def create_and_log_values(self, loss, outputs_argmax, labels, global_step, mode):
        # Calculate metrics
        f1 = f1_score(labels, outputs_argmax, average='macro')
        accuracy = accuracy_score(labels, outputs_argmax)
        precision = precision_score(labels, outputs_argmax, average='macro', zero_division=0)
        recall = recall_score(labels, outputs_argmax, average='macro')
        mae = mean_absolute_error(labels, outputs_argmax)
        try:
            auc_roc = roc_auc_score(labels, outputs_argmax, average='weighted', multi_class='ovr')
        except ValueError:
            auc_roc = 0.0
        mcc = matthews_corrcoef(labels, outputs_argmax)

        self.log_metrics(mode, global_step, Loss=loss, 
                         F1_Macro=f1, 
                         Accuracy=accuracy, 
                         Precision=precision, 
                         Recall=recall, 
                         Mae=mae, 
                         Auc_Roc=auc_roc, 
                         Mathews_corr=mcc)

        # print(f"Train Epoch {epoch}: Loss: {loss}, F1 Score: {f1}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, MAE: {mae}, AUC ROC: {auc_roc}, MCC: {mcc}")

    # def custom_log_validation(self, epoch, batch_idx, loss, outputs, labels):
    #     # Clone outputs and labels to avoid modifying the original tensors
    #     # ic(outputs)
    #     outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
    #     labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        
    #     # Apply softmax to the outputs and then take argmax
    #     outputs_softmax = F.softmax(torch.tensor(outputs), dim=1).numpy()
    #     # ic(outputs_softmax)
    #     outputs_argmax = np.argmax(outputs_softmax, axis=1)
    #     ic(outputs_argmax)
    #     ic(labels)

    #     # Calculate metrics
    #     f1 = f1_score(labels, outputs_argmax, average='weighted')
    #     accuracy = accuracy_score(labels, outputs_argmax)
    #     precision = precision_score(labels, outputs_argmax, average='weighted')
    #     recall = recall_score(labels, outputs_argmax, average='weighted')
    #     mae = mean_absolute_error(labels, outputs_argmax)
    #     try:
    #         auc_roc = roc_auc_score(labels, outputs_argmax, average='weighted', multi_class='ovr')
    #     except ValueError:
    #         auc_roc = 0.0
    #     mcc = matthews_corrcoef(labels, outputs_argmax)


    #     global_step = epoch * self.len_validation_loader + batch_idx

    #     # Log metrics for the current batch
    #     self.writer.add_scalar('Loss/Validation', loss, global_step=global_step)
    #     self.writer.add_scalar('F1_Score/Validation', f1, global_step=global_step)
    #     self.writer.add_scalar('Accuracy/Validation', accuracy, global_step=global_step)
    #     self.writer.add_scalar('Precision/Validation', precision, global_step=global_step)
    #     self.writer.add_scalar('Recall/Validation', recall, global_step=global_step)
    #     self.writer.add_scalar('MAE/Validation', mae, global_step=global_step )
    #     self.writer.add_scalar('AUC_ROC/Validation', auc_roc, global_step=global_step)
    #     self.writer.add_scalar('MCC/Validation', mcc, global_step=global_step)

    # def custom_log_test(self, batch_idx, loss, outputs, labels, epoch=0):
    #     # Clone outputs and labels to avoid modifying the original tensors
    #     outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
    #     labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    #     # Apply softmax to the outputs and then take argmax
    #     outputs_softmax = F.softmax(torch.tensor(outputs), dim=1).numpy()
    #     # ic(outputs_softmax)
    #     outputs_argmax = np.argmax(outputs_softmax, axis=1)

    #     global_step = epoch * self.len_test_loader + batch_idx

    #     self.create_scalar_values(loss, outputs_argmax, labels, global_step)

    # def custom_log_in_total(self, global_step, total_loss, outputs_all, labels_all, mode='validation'):
    #     """This is used to log the metrics for the entire validation dataset after all batches have been processed."""
    #     # Apply softmax to the outputs and then take argmax
    #     outputs_softmax = F.softmax(torch.tensor(outputs_all), dim=1).numpy()
    #     outputs_argmax = np.argmax(outputs_softmax, axis=1)
    #     self.create_scalar_values(total_loss, outputs_argmax, labels_all, global_step, mode=mode)


    # def log_additional_information(self, dataset_name: str, 
    #                      dataset_task: str, 
    #                      dataset_size: int, 
    #                      test_dataset_size: int,
    #                      validation_dataset_size: int,
    #                      train_dataset_size: int,
    #                      num_classes: int, 
    #                      optimizer_name: str, 
    #                      base_optimizer_name: str, 
    #                      learning_rate_of_base_optimizer: float, 
    #                      batch_size: int, 
    #                      epochs: int, 
    #                      k_approx: int, 
    #                      num_of_optimizer_iterations: int, 
    #                      seed_num: int,
    #                      range_to_select: int):
    #     # Initalize SummaryWriter for TensorBoard logging
    #     import datetime

    #     self.writer = SummaryWriter(log_dir=f"./runs/{dataset_task.upper()}_EPOCHS_{epochs}_FOSI_ITER_{num_of_optimizer_iterations}_TIME_{datetime.datetime.now().hour}:{datetime.datetime.now().minute}")  # Initialize SummaryWriter for TensorBoard logging

    #     # Log dataset information
    #     self.writer.add_text('Dataset Information', f'Dataset Name: {dataset_name}')
    #     self.writer.add_text('Dataset Information', f'Dataset Task: {dataset_task}')
    #     self.writer.add_text('Dataset Information', f'All Dataset Size: {dataset_size}')
    #     self.writer.add_text('Dataset Information', f'Test Dataset Size: {test_dataset_size}')
    #     self.writer.add_text('Dataset Information', f'Validation Dataset Size: {validation_dataset_size}')
    #     self.writer.add_text('Dataset Information', f'Train Dataset Size: {train_dataset_size}')
    #     self.writer.add_text('Dataset Information', f'Number of Classes: {num_classes}')
    #     self.writer.add_text('Dataset Information', f'Batch Size: {batch_size}')
    #     self.writer.add_text('Dataset Information', f'Epochs: {epochs}')
    #     self.writer.add_text('Optimization Information', f'Optimizer Name: {optimizer_name}')
    #     self.writer.add_text('Optimization Information', f'Base Optimizer Name: {base_optimizer_name}')
    #     self.writer.add_text('Optimization Information', f'Learning Rate of Base Optimizer: {learning_rate_of_base_optimizer}')
    #     self.writer.add_text('Optimization Information', f'Number of Max Eigenvalues to Approximate: {k_approx}')
    #     self.writer.add_text('Optimization Information', f'Number of Optimizer Iterations: {num_of_optimizer_iterations}')
    #     # if range to select is None then the entire dataset is used
    #     if range_to_select is None:
    #         range_to_select = 'All Dataset'
    #     self.writer.add_text('Optimization Information', f'Range to Select: {range_to_select}')
    #     self.writer.add_text('Optimization Information', f'Seed Number: {seed_num}')
        