import sys
import torch

from torch import nn


class SentenceDropout(nn.Module):
    def __init__(self, model_dim, dropout_probability):
        super().__init__()
        self.dim = model_dim
        self.p = dropout_probability

    def forward(self, x):
        num_dropout = int(x.shape[0]*self.p)
        random_indices = torch.randint(
            low=0, high=x.shape[0], size=(1, num_dropout))
        random_embeds = torch.rand(num_dropout, self.dim, 1, device='cuda')
        x[random_indices, :, :] = random_embeds
        return x


class SLPNet(nn.Module):
    def __init__(self, model_dim: int, num_classes: int, sentence_dropout: float, model_dropout: float):
        super().__init__()
        self.model_dim = model_dim
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if sentence_dropout > 0:
            self.dropout_layer = SentenceDropout(
                model_dim=model_dim, dropout_probability=sentence_dropout)
        else:
            self.dropout_layer = None

        self.embed_in = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=model_dim, kernel_size=1),
            nn.BatchNorm1d(model_dim)
        )

        self.encoder = nn.LSTM(input_size=model_dim,
                               hidden_size=model_dim,
                               dropout=model_dropout)

        self.classifier = nn.Linear(
            in_features=model_dim, out_features=num_classes)

    def forward(self, x, t):
        x = self.embed_in(x)  # (B, 1024, L) -> (B, model_dim, L)
        if self.dropout_layer != None:
            x = self.dropout_layer(x)  # (B, model_dim, L)
        x, _ = self.encoder(torch.transpose(x, 1, 2))  # -> (B, L, model_dim)
        x = self.classifier(x).squeeze()  # -> (B, num_classes)
        return x
