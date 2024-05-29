import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
 
class SOMetrics():
    def __init__(self, log_txt, output_path, epochs, dataset_length):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.all_metrics = np.zeros((epochs, 5))
        self.dataset_length = dataset_length
        self.output_path = output_path
        self.log_txt = log_txt
        
        self.metric_keys = ['PMR', 'Acc', 'KT', 'Rouge-S', 'LCS']
        self.pmr_score = 0
        self.acc_score = 0
        self.tau_score = 0
        self.r_s_score = 0
        self.lcs_score = 0
        
    def _perfect_match(self, ground_truth, predicted):
        self.pmr_score += torch.allclose(ground_truth, predicted)
    
    def _accuracy(self, ground_truth, predicted):
        self.acc_score += torch.sum(ground_truth == predicted)/len(ground_truth)

    def _kendall_tau(self, ground_truth, predicted):
        n = ground_truth.shape[0]
        sub_pairs = lambda x, n: (x.expand(n,n).T - x).sign_()
        self.tau_score += sub_pairs(ground_truth,n).mul_(sub_pairs(predicted,n)).sum().div(n*(n-1))
    
    def _ruoge_s(self):
        self.r_s_score = 0

    def _lcs(self):
        self.lcs_score = 0
    
    def batch(self, ground_truth, predicted):
        # select prediction
        _, predicted = torch.max(predicted, dim=1)
        
        # check inputs
        if ground_truth.dim() != 1 or predicted.dim() != 1:
            raise ValueError("Both inputs must be 1-dimensional tensors.")
    
        if len(ground_truth) != len(predicted):
            raise ValueError("Both tensors must have the same length.")
        
        # compute metrics 
        self._perfect_match(ground_truth, predicted)
        self._accuracy(ground_truth, predicted)
        self._kendall_tau(ground_truth, predicted)
        self._ruoge_s()
        self._lcs()
    
    def epoch_end(self, epoch):
        metric_values = [self.pmr_score, self.acc_score, 
                         self.tau_score, self.r_s_score, 
                         self.lcs_score]
        
        for i, (k, v) in enumerate(zip(self.metric_keys, metric_values)):
            self.all_metrics[epoch, i] = v / self.dataset_length
            utils.print_log(f"\t{k}\t: {v / self.dataset_length:.3}", self.log_txt, display=True)
        
        # reset memory
        self.pmr_score = 0
        self.acc_score = 0
        self.tau_score = 0
        self.rs_score = 0
        self.lcs_score = 0
    
    def eval_end(self, mode):        
        plt.figure(dpi=100, constrained_layout=True)
        for i, metric in enumerate(self.metric_keys):
            plt.plot(utils.remove_tailzeros(self.all_metrics[:, i]), label=metric)
            
        plt.xlabel('Epochs', fontsize=16)
        plt.legend(loc="upper left", fontsize=12)
        plt.savefig(self.output_path+f'results/{mode}.jpg')
        plt.close('all')
        