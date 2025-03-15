import math
import torch
from torch import nn


## Phase Recognition Model ##
class SentenceDropout(nn.Module):
    def __init__(self, model_dim, dropout_probability):
        super().__init__()
        self.dim = model_dim
        self.p = dropout_probability

    def forward(self, x):
        if self.training:
            mask = torch.ones_like(x)
            num_dropout = max(1, int(x.shape[0]*self.p))
            random_indices = torch.randint(
                low=0, high=x.shape[0], size=(1, num_dropout))
            mask[random_indices, :, :] = 0
            scale = (1 / (1 - self.p))
        elif not self.training:
            mask = torch.ones_like(x)
            scale = 1
        return x * mask * scale


class SLPNet(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_classes: int,
                 sentence_dropout: float,
                 model_dropout: float,
                 ):
        super().__init__()

        self.downsample = nn.Conv1d(
            in_channels=1024,
            out_channels=model_dim,
            kernel_size=1,
            bias=False
        )

        self.dropout = nn.Dropout(p=model_dropout)
        self.batchnorm = nn.BatchNorm1d(model_dim, affine=False)

        self.temporal_layer = nn.LSTM(input_size=model_dim,
                                      hidden_size=model_dim)

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(
            in_features=model_dim, out_features=num_classes)

    def forward(self, x):
        x_c = self.downsample(x)
        x_c = self.batchnorm(self.relu(x_c))
        x_c = self.dropout(x_c)

        x_c = torch.permute(x_c, (0,2,1))
        
        x_c, _ = self.temporal_layer(x_c)
        x_c = self.relu(x_c)

        x_c = self.classifier(x_c).squeeze()
        return x_c
