import torch
import utils
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class SPRDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_pickle(data_path)
        self.op_name = data_path.split('/')[-1][:-4]

    def __getitem__(self, index):
        return [self.data['Start_Time'].iloc[index],
                torch.FloatTensor(self.data['Embeddings'].iloc[index]),
                self.data['Phase_Label'].iloc[index]]

    def __len__(self):
        return len(self.data)


def get_dataset(data_path, log_txt, batch_size, synthetic_data_path=None):
    datasets = []
    data_splits = utils.data_split(data_path, 
                                   log_txt, 
                                   synthetic_data_path=synthetic_data_path)
    
    for data_list in data_splits:
        split_loaders = [
            DataLoader(
                dataset=SPRDataset(data_path),
                batch_size=batch_size,
                shuffle=False
            ) for data_path in data_list
        ]
        datasets.append(split_loaders)
        
    return datasets
