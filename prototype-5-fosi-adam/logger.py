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
        self.writer = SummaryWriter()  # Initialize SummaryWriter for TensorBoard logging

    def custom_log(self, epoch, batch_idx, loss, outputs, labels):
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
        f1 = f1_score(labels, outputs_argmax, average='weighted')
        accuracy = accuracy_score(labels, outputs_argmax)
        precision = precision_score(labels, outputs_argmax, average='weighted', zero_division=0)
        recall = recall_score(labels, outputs_argmax, average='weighted')
        mae = mean_absolute_error(labels, outputs_argmax)
        try:
            auc_roc = roc_auc_score(labels, outputs_argmax, average='weighted', multi_class='ovr')
        except ValueError:
            auc_roc = 0.0
        mcc = matthews_corrcoef(labels, outputs_argmax)


        global_step = epoch * self.len_train_loader + batch_idx

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

    def custom_log_validation(self, epoch, batch_idx, loss, outputs, labels):
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


        global_step = epoch * self.len_validation_loader + batch_idx

        # Log metrics for the current batch
        self.writer.add_scalar('Loss/Validation', loss, global_step=global_step)
        self.writer.add_scalar('F1_Score/Validation', f1, global_step=global_step)
        self.writer.add_scalar('Accuracy/Validation', accuracy, global_step=global_step)
        self.writer.add_scalar('Precision/Validation', precision, global_step=global_step)
        self.writer.add_scalar('Recall/Validation', recall, global_step=global_step)
        self.writer.add_scalar('MAE/Validation', mae, global_step=global_step )
        self.writer.add_scalar('AUC_ROC/Validation', auc_roc, global_step=global_step)
        self.writer.add_scalar('MCC/Validation', mcc, global_step=global_step)

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

    def close(self):
        self.writer.close() # Close the SummaryWriter
        