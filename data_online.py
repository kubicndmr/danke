import os
import torch
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

class SODataset(Dataset):
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