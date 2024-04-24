from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, roc_auc_score, matthews_corrcoef
import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic

class CustomLogger:
    def __init__(self, len_train_loader: int, len_validation_loader: int, len_test_loader: int) -> None:
        
        self.len_train_loader = len_train_loader
        self.len_validation_loader = len_validation_loader
        self.len_test_loader = len_test_loader

        self.train_losses = []
        self.train_f1_scores = []
        self.train_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_maes = []
        self.train_auc_roc = []
        self.train_mcc = []

        self.validation_losses = []
        self.validation_f1_scores = []
        self.validation_accuracies = []
        self.validation_precisions = []
        self.validation_recalls = []
        self.validation_maes = []
        self.validation_auc_roc = []
        self.validation_mcc = []

        self.test_losses = []
        self.test_f1_scores = []
        self.test_accuracies = []
        self.test_precisions = []
        self.test_recalls = []
        self.test_maes = []
        self.test_auc_roc = []
        self.test_mcc = []

    def custom_log(self, global_step, loss, outputs, labels):
        # Clone outputs and labels to avoid modifying the original tensors
        # ic(outputs)
        outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
        labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        # Apply softmax to the outputs and then take argmax
        outputs_softmax = F.softmax(torch.tensor(outputs), dim=1).numpy()
        # ic(outputs_softmax)
        outputs_argmax = np.argmax(outputs_softmax, axis=1)
        ic(outputs_argmax)
        ic(labels)

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

        # Log metrics for the current batch
        self.writer.add_scalar('Loss/Train', loss, global_step=global_step)
        self.writer.add_scalar('F1_Score/Train', f1, global_step=global_step)
        self.writer.add_scalar('Accuracy/Train', accuracy, global_step=global_step)
        self.writer.add_scalar('Precision/Train', precision, global_step=global_step)
        self.writer.add_scalar('Recall/Train', recall, global_step=global_step)
        self.writer.add_scalar('MAE/Train', mae, global_step=global_step )
        self.writer.add_scalar('AUC_ROC/Train', auc_roc, global_step=global_step)
        self.writer.add_scalar('MCC/Train', mcc, global_step=global_step)

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

    def custom_log_test(self, batch_idx, loss, outputs, labels, epoch=0):
        # Clone outputs and labels to avoid modifying the original tensors
        # ic(outputs)
        outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
        labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        
        # Apply softmax to the outputs and then take argmax
        outputs_softmax = F.softmax(torch.tensor(outputs), dim=1).numpy()
        # ic(outputs_softmax)
        outputs_argmax = np.argmax(outputs_softmax, axis=1)
        ic(outputs_argmax)
        ic(labels)

        global_step = epoch * self.len_test_loader + batch_idx

        # Calculate metrics
        f1 = f1_score(labels, outputs_argmax, average='weighted')
        accuracy = accuracy_score(labels, outputs_argmax)
        precision = precision_score(labels, outputs_argmax, average='weighted')
        recall = recall_score(labels, outputs_argmax, average='weighted')
        mae = mean_absolute_error(labels, outputs_argmax)
        try:
            auc_roc = roc_auc_score(labels, outputs_argmax, average='weighted', multi_class='ovr')
        except ValueError:
            auc_roc = 0.0
        mcc = matthews_corrcoef(labels, outputs_argmax)

        # Log metrics for the current batch
        self.writer.add_scalar('Loss/Test', loss, global_step=global_step)
        self.writer.add_scalar('F1_Score/Test', f1, global_step=global_step)
        self.writer.add_scalar('Accuracy/Test', accuracy, global_step=global_step)
        self.writer.add_scalar('Precision/Test', precision, global_step=global_step)
        self.writer.add_scalar('Recall/Test', recall, global_step=global_step)
        self.writer.add_scalar('MAE/Test', mae, global_step=global_step)
        self.writer.add_scalar('AUC_ROC/Test', auc_roc, global_step=global_step)
        self.writer.add_scalar('MCC/Test', mcc, global_step=global_step)

    def custom_log_in_total(self, global_step, total_loss, outputs_all, labels_all, mode='validation'):
        """This is used to log the metrics for the entire validation dataset after all batches have been processed."""

        # Clone outputs and labels to avoid modifying the original tensors
        outputs_all = [output.clone().detach().cpu().numpy() for output in outputs_all]
        labels_all = [label.clone().detach().cpu().numpy() for label in labels_all]
        
        # Convert outputs and labels to NumPy arrays
        outputs_all = np.concatenate(outputs_all)
        labels_all = np.concatenate(labels_all)

        # Apply softmax to the outputs and then take argmax
        outputs_softmax = F.softmax(torch.tensor(outputs_all), dim=1).numpy()
        outputs_argmax = np.argmax(outputs_softmax, axis=1)

        # Calculate metrics
        f1 = f1_score(labels_all, outputs_argmax, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels_all, outputs_argmax)
        precision = precision_score(labels_all, outputs_argmax, average='weighted', zero_division=0)
        recall = recall_score(labels_all, outputs_argmax, average='weighted', zero_division=0)
        mae = mean_absolute_error(labels_all, outputs_argmax)
        auc_roc = roc_auc_score(labels_all, outputs_argmax, average='weighted', multi_class='ovr')
        mcc = matthews_corrcoef(labels_all, outputs_argmax)

        global_step = global_step

        # Log mean metrics for all batches
        self.writer.add_scalar(f'Loss/{mode.capitalize()}', total_loss, global_step=global_step)
        self.writer.add_scalar(f'F1_Score/{mode.capitalize()}', f1, global_step=global_step)
        self.writer.add_scalar(f'Accuracy/{mode.capitalize()}', accuracy, global_step=global_step)
        self.writer.add_scalar(f'Precision/{mode.capitalize()}', precision, global_step=global_step)
        self.writer.add_scalar(f'Recall/{mode.capitalize()}', recall, global_step=global_step)
        self.writer.add_scalar(f'MAE/{mode.capitalize()}', mae, global_step=global_step)
        self.writer.add_scalar(f'AUC_ROC/{mode.capitalize()}', auc_roc, global_step=global_step)
        self.writer.add_scalar(f'MCC/{mode.capitalize()}', mcc, global_step=global_step)


    def log_additional_information(self, dataset_name: str, 
                         dataset_task: str, 
                         dataset_size: int, 
                         test_dataset_size: int,
                         validation_dataset_size: int,
                         train_dataset_size: int,
                         num_classes: int, 
                         optimizer_name: str, 
                         base_optimizer_name: str, 
                         learning_rate_of_base_optimizer: float, 
                         batch_size: int, 
                         epochs: int, 
                         k_approx: int, 
                         num_of_optimizer_iterations: int, 
                         seed_num: int,
                         range_to_select: int):
        # Initalize SummaryWriter for TensorBoard logging
        import datetime

        self.writer = SummaryWriter(log_dir=f"./runs/{dataset_task.upper()}_EPOCHS_{epochs}_FOSI_ITER_{num_of_optimizer_iterations}_TIME_{datetime.datetime.now().hour}:{datetime.datetime.now().minute}")  # Initialize SummaryWriter for TensorBoard logging

        # Log dataset information
        self.writer.add_text('Dataset Information', f'Dataset Name: {dataset_name}')
        self.writer.add_text('Dataset Information', f'Dataset Task: {dataset_task}')
        self.writer.add_text('Dataset Information', f'All Dataset Size: {dataset_size}')
        self.writer.add_text('Dataset Information', f'Test Dataset Size: {test_dataset_size}')
        self.writer.add_text('Dataset Information', f'Validation Dataset Size: {validation_dataset_size}')
        self.writer.add_text('Dataset Information', f'Train Dataset Size: {train_dataset_size}')
        self.writer.add_text('Dataset Information', f'Number of Classes: {num_classes}')
        self.writer.add_text('Dataset Information', f'Batch Size: {batch_size}')
        self.writer.add_text('Dataset Information', f'Epochs: {epochs}')
        self.writer.add_text('Optimization Information', f'Optimizer Name: {optimizer_name}')
        self.writer.add_text('Optimization Information', f'Base Optimizer Name: {base_optimizer_name}')
        self.writer.add_text('Optimization Information', f'Learning Rate of Base Optimizer: {learning_rate_of_base_optimizer}')
        self.writer.add_text('Optimization Information', f'Number of Max Eigenvalues to Approximate: {k_approx}')
        self.writer.add_text('Optimization Information', f'Number of Optimizer Iterations: {num_of_optimizer_iterations}')
        # if range to select is None then the entire dataset is used
        if range_to_select is None:
            range_to_select = 'All Dataset'
        self.writer.add_text('Optimization Information', f'Range to Select: {range_to_select}')
        self.writer.add_text('Optimization Information', f'Seed Number: {seed_num}')


    def close(self):
        self.writer.close() # Close the SummaryWriter
        