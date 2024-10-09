import os
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


class SPRMetrics:
    def __init__(self, log_txt, output_path, epochs):
        self.metric_keys = ['Acc', 'F1', 'Recall', 'Precision', 'Jaccard']
        self.real_metrics = np.zeros((epochs, len(self.metric_keys)))
        self.syn_metrics = np.zeros((epochs, len(self.metric_keys)))
        self.all_metrics = np.zeros((epochs, len(self.metric_keys)))
        self.output_path = output_path
        self.log_txt = log_txt

        self.epoch = 0
        self.op_gt = []
        self.op_pr = []
        self.op_metrics = []

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
            'Acc': accuracy_score(ground_truth, prediction),
            'F1': f1_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Recall': recall_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Precision': precision_score(ground_truth, prediction, average='macro', zero_division=0.0),
            'Jaccard': jaccard_score(ground_truth, prediction, average='macro', zero_division=0.0)
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

    def epoch_end(self, epoch):
        real = len(
            [1 for metrics in self.op_metrics if metrics['Name'].startswith('Real')]) > 0
        syn = len(
            [1 for metrics in self.op_metrics if metrics['Name'].startswith('Syn')]) > 0

        # Log metrics to file
        for i, metric in enumerate(self.metric_keys):
            utils.print_log(f"\t{metric}", self.log_txt)
            avg_metric = np.mean([metrics[metric]
                                 for metrics in self.op_metrics])
            for metrics in self.op_metrics:
                utils.print_log(f"\t\t{metrics['Name']}\t: {metrics[metric]:.3f}",
                                self.log_txt)
            utils.print_log(f"\t\tMean {metric}:\t {avg_metric:.3f}",
                            self.log_txt,
                            display=True)

            self.all_metrics[epoch, i] = avg_metric

            if real:
                self.real_metrics[epoch, i] = np.mean(
                    [metrics[metric] for metrics in self.op_metrics if metrics['Name'].startswith('Real')])
            if syn:
                self.syn_metrics[epoch, i] = np.mean(
                    [metrics[metric] for metrics in self.op_metrics if metrics['Name'].startswith('Syn')])

        # Reset memory
        self.op_metrics = []
        self.epoch += 1

    def eval_end(self, mode):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 18

        real = np.any(self.real_metrics)
        syn = np.any(self.syn_metrics)

        # All
        if not (real or syn):
            plt.figure(dpi=100, constrained_layout=True)
            for i, metric in enumerate(self.metric_keys):
                plt.plot(utils.remove_tailzeros(
                    self.all_metrics[:, i]), label=metric)
                utils.print_log(f"[{mode}-All]\tMax {metric} is: {np.max(self.all_metrics[:, i]):.5f} Epoch: {np.argmax(self.all_metrics[:, i])}",
                                self.log_txt,
                                display=True)

            plt.xlabel('Epochs', fontsize=18)
            plt.legend(loc="upper left", fontsize=12)
            plt.savefig(self.output_path+f'results/{mode}_metrics.jpg')

        # Real
        if real:
            plt.figure(dpi=100, constrained_layout=True)
            for i, metric in enumerate(self.metric_keys):
                plt.plot(utils.remove_tailzeros(
                    self.real_metrics[:, i]), label=metric)
                utils.print_log(f"[{mode}-Real]\tMax {metric} is: {np.max(self.real_metrics[:, i]):.5f} Epoch: {np.argmax(self.real_metrics[:, i])}",
                                self.log_txt,
                                display=True)

            plt.xlabel('Epochs', fontsize=18)
            plt.legend(loc="upper left", fontsize=12)
            plt.savefig(self.output_path+f'results/{mode}_real_metrics.jpg')

        # Synthetic
        if syn:
            plt.figure(dpi=100, constrained_layout=True)
            for i, metric in enumerate(self.metric_keys):
                plt.plot(utils.remove_tailzeros(
                    self.syn_metrics[:, i]), label=metric)
                utils.print_log(f"[{mode}-Syn]\tMax {metric} is: {np.max(self.syn_metrics[:, i]):.5f} Epoch: {np.argmax(self.syn_metrics[:, i])}",
                                self.log_txt,
                                display=True)

            plt.xlabel('Epochs', fontsize=18)
            plt.legend(loc="upper left", fontsize=12)
            plt.savefig(self.output_path+f'results/{mode}_syn_metrics.jpg')

        plt.close('all')
