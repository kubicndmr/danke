import torch.nn.functional as F

from torch import nn
from torch import Tensor
from transformers import BertModel
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        if config["llm"] == "bert":
            self.llm = BertModel.from_pretrained(config["model_name"])
        elif config["llm"] == "e5-large":
            self.llm = AutoModel.from_pretrained(config["model_name"])
        else:
            NotImplementedError

        if config["freeze"]:
            assert config["freeze_layers"] >= 0 and config["freeze_layers"] <= 23, "Can freeze up to 20 blocks"
            for name, param in self.llm.named_parameters():
                if name.startswith("embeddings"):
                    param.requires_grad = False
                if any(f"encoder.layer.{i}." in name for i in range(config["freeze_layers"])):
                    param.requires_grad = False

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask):
        output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        if self.config["llm"] == "bert":
            return output.pooler_output
        elif self.config["llm"] == "e5-large":
            embeddings = self.average_pool(
                output.last_hidden_state, attention_mask)
            return F.normalize(embeddings, p=2, dim=1)


class TextClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.head = config["head"]

        if self.head == "single_layer":
            self.classifier = nn.Sequential(
                nn.Dropout(p=config["ff_dropout"]),
                nn.Conv1d(
                    in_channels=config["input_dim"],
                    out_channels=config["n_classes"],
                    kernel_size=1,
                )
            )

        elif self.head == "temporal":
            self.dropout = nn.Dropout(p=config["ff_dropout"])

            self.conv1 = nn.Conv1d(
                in_channels=config["input_dim"],
                out_channels=config["model_dim"],
                kernel_size=1,
            )

            self.bn1 = nn.BatchNorm1d(config["model_dim"])
            self.relu = nn.ReLU()

            self.lstm = nn.LSTM(
                input_size=config["model_dim"],
                hidden_size=config["model_dim"],
                batch_first=True,   # IMPORTANT
            )

            self.conv_out = nn.Conv1d(
                in_channels=config["model_dim"],
                out_channels=config["n_classes"],
                kernel_size=1,
            )

        else:
            raise NotImplementedError

    def forward(self, x):
        x = x.permute(1, 0)  # (encoder_dim, B)

        if self.head == "single_layer":
            x = self.classifier(x)  # (n_classes, B)
            x = x.permute(1, 0)  # (B, n_classes
            return x

        elif self.head == "temporal":
            x = self.dropout(x)

            x = self.conv1(x) # (model_dim, B)
            x = x.permute(1, 0)
            x = self.bn1(x) # (B, model_dim)
            x = self.relu(x)

            x, _ = self.lstm(x) # (B, lstm_dim)

            x = x.permute(1, 0) # (lstm_dim, B)
            x = self.conv_out(x) # (n_classes, B)
            x = x.permute(1, 0) # (B, n_classes)
            return x


class SLPNet(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()

        # Text encoders
        self.text_encoder = TextEncoder(config.embedder_config)

        # Classifier
        self.classifier = TextClassifier(config.classifier_config)

    def forward(self, x, attmask): # x: (B, T)
        x = self.text_encoder(x, attmask) # (B, C=1024)
        x = self.classifier(x) # (B, n_classes)
        return x
