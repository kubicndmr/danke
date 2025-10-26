import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import SurgPhaseRecog.utils as utils

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

class SPRMetrics:
    def __init__(self, log_txt, output_path, epochs):
        self.metric_keys = ['Accuracy', 'F1_Score',
                            'Recall', 'Precision', 'Jaccard']
        self.metrics = np.zeros((epochs, len(self.metric_keys)))
        self.output_path = output_path
        self.log_txt = log_txt

        self.epoch = 0
        self.op_gt = []
        self.op_pr = []
        self.op_metrics = []
        self.confusion_matrices = []

    def batch(self, ground_truth, predicted):
        # Select prediction
        _, predicted = torch.max(predicted, dim=1)

        # Validate inputs
        if ground_truth.dim() != 1 or predicted.dim() != 1:
            raise ValueError("Both inputs must be 1-dimensional tensors.")

        if len(ground_truth) != len(predicted):
            raise ValueError("Both tensors must have the same length.")

        self.op_gt.append(ground_truth.cpu().numpy())
        self.op_pr.append(predicted.cpu().numpy())

    def op_end(self, op_name, plot_ribbon=False):
        # Convert lists to numpy arrays
        ground_truth = np.concatenate(self.op_gt)
        prediction = np.concatenate(self.op_pr)

        # Exclude transition periods
        transition_indices = np.where(ground_truth == 8)
        ground_truth = np.delete(ground_truth, transition_indices)
        prediction = np.delete(prediction, transition_indices)

        # Compute metrics
        metrics = {
            'Name': op_name,
            'Accuracy': accuracy_score(ground_truth, prediction),
            'F1_Score': f1_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Recall': recall_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Precision': precision_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Jaccard': jaccard_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Confusion_Matrix': confusion_matrix(ground_truth, prediction, labels=range(8)),
        }
        self.op_metrics.append(metrics)

        # Plot ribbon
        if plot_ribbon:
            save_dir = self.output_path + \
                f"results/ribbons/{op_name}/"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            utils.plot_ribbon(prediction, f'Epoch_{self.epoch}', save_dir)

        # Reset memory
        self.op_gt = []
        self.op_pr = []

    def epoch_end(self, epoch: int, log_opwise_metrics: bool):
        # Log metrics to file
        for i, metric in enumerate(self.metric_keys):
            avg_metric = np.mean([metrics[metric]
                                 for metrics in self.op_metrics])
            if log_opwise_metrics:
                for metrics in self.op_metrics:
                    utils.print_log(f"\t\t{metrics['Name']}\t: {metrics[metric]:.3f}",
                                    self.log_txt)
            utils.print_log(f"\tMean {metric}\t: {avg_metric:.3f}",
                            self.log_txt,
                            display=True)

            self.metrics[epoch, i] = avg_metric

        # Get list of confusion matrices
        self.confusion_matrices.append([metrics['Confusion_Matrix']
                                   for metrics in self.op_metrics])

        # Reset memory
        self.op_metrics = []
        self.epoch += 1

    def eval_end(self, mode):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        max_values = []

        # Plot
        plt.figure(dpi=100, constrained_layout=True)
        for i, metric in enumerate(self.metric_keys):
            metric_values = self.metrics[:, i]
            max_value = np.max(metric_values)
            max_epoch = np.argmax(metric_values)
            max_values.append(max_value)

            utils.print_log(f"[{mode}]\tMax {metric} is: {max_value:.5f} Epoch: {max_epoch}",
                            self.log_txt,
                            display=True)
            
            plt.plot(utils.remove_tailzeros(metric_values), label=metric)
            
            if metric == 'Accuracy':
                best_epoch = max_epoch

        plt.xlabel('Epochs', fontsize=18)
        plt.legend(loc="upper left", fontsize=12)
        plt.savefig(self.output_path+f'results/{mode}_metrics.jpg')
        plt.close('all')

        return max_values, self.confusion_matrices[best_epoch]