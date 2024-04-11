import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_absolute_error, roc_auc_score, matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging

class Logger:
    def __init__(self):
        self.train_losses = []
        self.train_f1_scores = []
        self.train_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_maes = []
        self.train_auc_roc = []
        self.train_mcc = []  # Added list for MCC
        self.validation_losses = []  # Store validation losses
        self.validation_f1_scores = []  # Store validation F1 scores
        self.writer = SummaryWriter()  # Initialize SummaryWriter for TensorBoard logging

    def log_train_epoch(self, epoch, loss, preds, labels):
        self.train_losses.append(loss)

        train_f1 = f1_score(labels, preds)
        self.train_f1_scores.append(train_f1)

        train_accuracy = accuracy_score(labels, preds)
        self.train_accuracies.append(train_accuracy)

        train_precision = precision_score(labels, preds)
        self.train_precisions.append(train_precision)

        train_recall = recall_score(labels, preds)
        self.train_recalls.append(train_recall)

        train_mae = mean_absolute_error(labels, preds)
        self.train_maes.append(train_mae)

        train_auc_roc = roc_auc_score(labels, preds)
        self.train_auc_roc.append(train_auc_roc)

        train_mcc = matthews_corrcoef(labels, preds)  # Calculate MCC
        self.train_mcc.append(train_mcc)  # Append MCC to the list

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('F1 Score/train', train_f1, epoch)
        self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        self.writer.add_scalar('Precision/train', train_precision, epoch)
        self.writer.add_scalar('Recall/train', train_recall, epoch)
        self.writer.add_scalar('MAE/train', train_mae, epoch)
        self.writer.add_scalar('AUC ROC/train', train_auc_roc, epoch)
        self.writer.add_scalar('MCC/train', train_mcc, epoch)  # Log MCC to TensorBoard

        print(f"Epoch {epoch}: Loss: {loss}, F1 Score: {train_f1}, Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, MAE: {train_mae}, AUC ROC: {train_auc_roc}, MCC: {train_mcc}")

    def log_validation_epoch(self, epoch, loss, preds, labels):
        self.validation_losses.append(loss)

        validation_f1 = f1_score(labels, preds)
        self.validation_f1_scores.append(validation_f1)

        validation_accuracy = accuracy_score(labels, preds)
        validation_precision = precision_score(labels, preds)
        validation_recall = recall_score(labels, preds)
        validation_mae = mean_absolute_error(labels, preds)
        validation_auc_roc = roc_auc_score(labels, preds)
        validation_mcc = matthews_corrcoef(labels, preds)

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/validation', loss, epoch)
        self.writer.add_scalar('F1 Score/validation', validation_f1, epoch)
        self.writer.add_scalar('Accuracy/validation', validation_accuracy, epoch)
        self.writer.add_scalar('Precision/validation', validation_precision, epoch)
        self.writer.add_scalar('Recall/validation', validation_recall, epoch)
        self.writer.add_scalar('MAE/validation', validation_mae, epoch)
        self.writer.add_scalar('AUC ROC/validation', validation_auc_roc, epoch)
        self.writer.add_scalar('MCC/validation', validation_mcc, epoch)

        print(f"Validation at epoch {epoch}: Loss: {loss}, F1 Score: {validation_f1}, Accuracy: {validation_accuracy}, Precision: {validation_precision}, Recall: {validation_recall}, MAE: {validation_mae}, AUC ROC: {validation_auc_roc}, MCC: {validation_mcc}")

    def log_test_metrics(self, test_loss, test_preds, test_labels):
        test_f1 = f1_score(test_labels, test_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)
        test_precision = precision_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds)
        test_mae = mean_absolute_error(test_labels, test_preds)
        test_auc_roc = roc_auc_score(test_labels, test_preds)
        test_mcc = matthews_corrcoef(test_labels, test_preds)

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/test', test_loss)
        self.writer.add_scalar('F1 Score/test', test_f1)
        self.writer.add_scalar('Accuracy/test', test_accuracy)
        self.writer.add_scalar('Precision/test', test_precision)
        self.writer.add_scalar('Recall/test', test_recall)
        self.writer.add_scalar('MAE/test', test_mae)
        self.writer.add_scalar('AUC ROC/test', test_auc_roc)
        self.writer.add_scalar('MCC/test', test_mcc)

        print(f"Test Metrics: Loss: {test_loss}, F1 Score: {test_f1}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, MAE: {test_mae}, AUC ROC: {test_auc_roc}, MCC: {test_mcc}")

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 5, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.validation_losses, label='Validation Loss')  # Add validation loss plot
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 5, 2)
        plt.plot(epochs, self.train_f1_scores, label='Training F1 Score')
        plt.plot(epochs, self.validation_f1_scores, label='Validation F1 Score')  # Add validation F1 score plot
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()

        plt.show()

    def close(self):
        self.writer.close()  # Close the SummaryWriter when logging is complete
