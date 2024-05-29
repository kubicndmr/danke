import os
import torch
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

class SODatasetOnline(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
        self.model.to(self.device)


    def __getitem__(self, index):
        row = self.data.iloc[index]
        shuffled_embeddings = self._get_embeddings(row['ShuffledSentences'])
        shuffled_indices = torch.tensor(row['ShuffledIndices'])
        return (shuffled_embeddings, shuffled_indices)

    def __len__(self):
        return len(self.data)
    
    def _get_embeddings(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True).T
    

class SODatasetOffline(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        shuffled_embeddings = row['ShuffledEmbeddings']
        shuffled_indices = torch.tensor(row['ShuffledIndices'])
        return (shuffled_embeddings, shuffled_indices)

    def __len__(self):
        return len(self.data)


def get_dataset(data_path, mode):
    """
    Loads datasets based on the specified mode ('online' or 'offline').

    Parameters:
    - data_path (str): The path to the directory containing the data files.
    - mode (str): The mode in which to load the data ('online' or 'offline').

    Returns:
    - datasets (list): A list containing DataLoader objects for train, test, and validation datasets.
    """
    
    datasets = []

    if mode == 'online':
        # Load online datasets
        for split in ['train', 'test', 'valid']:
            dataset = SODatasetOnline(os.path.join(data_path, f'{split}.pkl'))
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            datasets.append(loader)
    
    elif mode == 'offline':
        # Load offline datasets
        for split in ['train', 'test', 'valid']:
            dataset = SODatasetOffline(os.path.join(data_path, f'{split}.pkl'))
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            datasets.append(loader)
        
    else:
        raise ValueError("Mode must be either 'online' or 'offline'")

    return datasets