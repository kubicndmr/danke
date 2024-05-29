import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class SODataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        shuffled_embeddings = row['ShuffledEmbeddings']
        shuffled_indices = torch.tensor(row['ShuffledIndices'])
        return (shuffled_embeddings, shuffled_indices)

    def __len__(self):
        return len(self.data)

def get_dataset(data_path):
    datasets = list()
    
    for split in ['train', 'test', 'valid']:
        datasets.append(
            DataLoader(
                dataset=SODataset(os.path.join(data_path, split + '.pkl')),
                batch_size=1,
                shuffle=False
            )
        )
        
    return datasets