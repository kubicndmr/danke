import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class SPRDataset(Dataset):
    def __init__(self, data_path, batch_size):
        self.data = self.get_data(data_path, batch_size)
        self.op_name = data_path.split('/')[-1][:-4]

    def __getitem__(self, index):
        return [self.data.index[index],
                torch.FloatTensor(self.data['Embeddings'].iloc[index]),
                self.data['Phase_Label'].iloc[index]]

    def get_data(self, data_path, batch_size):
        df = pd.read_pickle(data_path)
        df = df[df['Phase_Label'] != 8]
        if len(df) % batch_size == 1:
            df = df.iloc[:-1]
        return df

    def phase_count(self):
        array_count = np.zeros(8, dtype=int)
        phases = self.data['Phase_Label'].astype(int).value_counts()
        array_count[phases.index] = phases.values
        return array_count

    def __len__(self):
        return len(self.data)


def get_dataset(data_list: list, batch_size: int):
    data_loaders = [
        DataLoader(
            dataset=SPRDataset(data_path, batch_size),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        ) for data_path in data_list
    ]

    data_size = np.sum([d_l.dataset.__len__() for d_l in data_loaders])
    data_batchsize = np.sum(
        [1 for data_loader in data_loaders for _, _, _ in data_loader])

    return {'data': data_loaders,
            'size': data_size,
            'batch_size': data_batchsize}
