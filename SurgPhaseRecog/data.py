import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class SPRDataset(Dataset):
    def __init__(self, data_path, config):
        # Data path
        self.config = config
        self.data_path = data_path
        self.op_name = data_path.split('/')[-1][:-4]

        # Tokenizer
        if config["llm"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                config["model_name"])
        elif config["llm"] == "e5-large":
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"])
        else:
            raise NotImplementedError
        
        # Extract data
        self.get_data(data_path, config["batch_size"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return [
            self.input_ids[index],
            self.attention_mask[index],
            self.labels[index]
        ]

    def get_data(self, data_path, batch_size):
        # Read
        df = pd.read_csv(data_path, index_col=0)

        # Remove transition label
        df = df[df['Phase_Label'] != 8].reset_index(drop=True)

        # Add query
        if self.config['llm'] == 'e5-large':
            df['Text'] = df['Text'].apply(lambda x: 'query: ' + x)

        # If batch size is 1, remove last sample
        if len(df) % batch_size == 1:
            df = df.iloc[:-1].reset_index(drop=True)

        # Tokenize text
        inputs = self.tokenizer(
            df['Text'].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            return_attention_mask=True
        )

        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']

        # Labels as a tensor
        self.labels = torch.tensor(df['Phase_Label'].values, dtype=torch.long)
        
        # Compute phaase lengths
        self.phase_count = np.zeros(8, dtype=int)
        phases = df['Phase_Label'].astype(int).value_counts()
        self.phase_count[phases.index] = phases.values

    def __len__(self):
        return len(self.labels)

    def op_type(self):
        if self.op_name.startswith('Real'):
            return 'real'
        elif self.op_name.startswith('Syn'):
            return 'syn'


def get_dataset(data_list: list, config: dict):

    data_loaders = [
        DataLoader(
            dataset=SPRDataset(data_path, config),
            batch_size=config["batch_size"],
            shuffle=False,
            pin_memory=True
        ) for data_path in data_list
    ]

    data_size = np.sum([d_l.dataset.__len__() for d_l in data_loaders])
    data_batchsize = np.sum(
        [1 for data_loader in data_loaders for _, _, _ in data_loader])

    phase_counts = np.zeros(8, dtype=int)
    for dl in data_loaders:
        phase_counts += dl.dataset.phase_count

    return {
        'data': data_loaders,
        'size': data_size,
        'batch_size': data_batchsize,
        'phase_counts': phase_counts
    }
