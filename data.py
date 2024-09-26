import torch
import utils
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
        if len(df) % batch_size == 1:
            df = df.iloc[:-1]
        return df

    def phase_count(self):
        array_count = np.zeros(8, dtype=int)
        phases = self.data['Phase_Label'].astype(int).value_counts().drop(
            8, errors='ignore')
        array_count[phases.index] = phases.values
        return array_count

    def __len__(self):
        return len(self.data)


def get_dataset(data_path, train_mode, log_txt, batch_size, num_train_ops=None):
    datasets = []
    data_splits = utils.data_split(data_path,
                                   train_mode,
                                   log_txt,
                                   num_train_ops
                                   )

    for data_list in data_splits:
        split_loaders = [
            DataLoader(
                dataset=SPRDataset(data_path, batch_size),
                batch_size=batch_size,
                shuffle=False
            ) for data_path in data_list
        ]
        datasets.append(split_loaders)

    return datasets
