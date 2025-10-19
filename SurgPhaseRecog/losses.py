import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WCELoss(nn.Module):
    def __init__(self, weight=None):
        super(WCELoss, self).__init__()
        self.weight = weight

    def forward(self, predict, label):
        return F.cross_entropy(
            predict, label,
            weight=self.weight,
            reduction='mean',
            ignore_index=8
        )


class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, predict, label):
        probs = F.softmax(predict, dim=1)
        targets = F.one_hot(label, num_classes=8).float()
        
        ce_loss = -targets * torch.log(probs)
        
        p_t = torch.sum(probs * targets, dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight.unsqueeze(1) * ce_loss
        return loss.mean()


class LDAMLoss(nn.Module):    
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, device='cuda')
        self.m_list = m_list
        assert s > 0
        self.s = s
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.weight = weight.to(self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s*output, target, weight=self.weight, ignore_index = 8)
