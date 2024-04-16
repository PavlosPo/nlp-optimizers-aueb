from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, roc_auc_score, matthews_corrcoef
import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic

class CustomLogger:
    def __init__(self, len_train_loader: int) -> None:
        
        self.len_train_loader = len_train_loader

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
        self.writer = SummaryWriter()

    def custom_log(self, epoch, batch_idx, loss, outputs, labels):
        # Clone outputs and labels to avoid modifying the original tensors
        ic(outputs)
        outputs = outputs.clone().detach().cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
        labels = labels.clone().detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
        # Apply softmax to the outputs and then take argmax
        outputs_softmax = F.softmax(torch.tensor(outputs), dim=1).numpy()
        ic(outputs_softmax)
        outputs_argmax = np.argmax(outputs_softmax, axis=1)
        ic(outputs_argmax)
        ic(labels)

        # Calculate metrics
        f1 = f1_score(labels, outputs_argmax, average='weighted')
        accuracy = accuracy_score(labels, outputs_argmax)
        precision = precision_score(labels, outputs_argmax, average='weighted')
        recall = recall_score(labels, outputs_argmax, average='weighted')
        mae = mean_absolute_error(labels, outputs_argmax)
        auc_roc = roc_auc_score(labels, outputs_argmax, average='weighted', multi_class='ovr')
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
