import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class SLPNet(nn.Module):
    def __init__(self, model_dim, num_stages, num_layers, num_classes):
        super().__init__()
        
        self.embed_in = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=model_dim, 
                kernel_size=1
            ),
            nn.BatchNorm1d(model_dim)
        )
        
        temporal_model = OrderedDict()
        for n in range(num_stages):
            temporal_model[f'TCN{n}'] = SingleStageModel(
                num_layers, 
                model_dim,
                num_classes, 
                causal_conv=True
            )
        self.temporal_model = nn.Sequential(temporal_model)
                
    def forward(self, x):
        x = self.embed_in(x)
        
        x = self.temporal_model(torch.transpose(x, 1, 2))
        return x