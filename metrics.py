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
        self.all_metrics = np.zeros((epochs, len(self.metric_keys)))
        self.output_path = output_path
        self.log_txt = log_txt
        
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

    def op_end(self, op_name):
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
        
        # Reset memory
        self.op_gt = []
        self.op_pr = []

    def epoch_end(self, epoch):
        # Log metrics to file
        for i, metric in enumerate(self.metric_keys):
            utils.print_log(f"\t{metric}", self.log_txt)
            avg_metric = np.mean([metrics[metric] for metrics in self.op_metrics])
            for metrics in self.op_metrics:
                utils.print_log(f"\t\t{metrics['Name']}\t: {metrics[metric]:.3f}", 
                                self.log_txt)
            utils.print_log(f"\t\tMean {metric}:\t {avg_metric:.3f}", 
                            self.log_txt,
                            display=True)
            self.all_metrics[epoch, i] = avg_metric

        # Reset memory
        self.op_metrics = []
    
    def eval_end(self, mode):        
        plt.figure(dpi=100, constrained_layout=True)
        for i, metric in enumerate(self.metric_keys):
            plt.plot(utils.remove_tailzeros(self.all_metrics[:, i]), label=metric)
            
        plt.xlabel('Epochs', fontsize=16)
        plt.legend(loc="upper left", fontsize=12)
        plt.savefig(self.output_path+f'results/{mode}_metrics.jpg')
        plt.close('all')
        