import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

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

        if sentence_dropout > 0:
            self.sentence_dropout_layer = SentenceDropout(
                model_dim=model_dim, dropout_probability=sentence_dropout)
        else:
            self.sentence_dropout_layer = None

        self.positional_encoder = PositionalEncoding(
            d_model=model_dim,
            dropout=model_dropout,
            max_len=512
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

        if self.training and self.sentence_dropout_layer:
            x_c = self.sentence_dropout_layer(x_c)

        x_c = torch.permute(x_c, (0,2,1))
        x_c = self.positional_encoder(x_c)
        
        x_c, _ = self.temporal_layer(x_c)
        x_c = self.relu(x_c)

        x_c = self.classifier(x_c).squeeze()
        return x_c
