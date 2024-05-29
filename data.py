import json
import utils
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class OPDataset(Dataset):
    def __init__(self, data_path, annot_path, model_dim):
        self.data_path = data_path
        self.annot_path = annot_path
        self.model_dim = model_dim
        self.data = pd.read_pickle(data_path)
        self.duration = self._get_duration()
        self.phase_intervals = self._get_intervals()

    def __getitem__(self, index):
        time_start = self.positional_encoding(
            self.model_dim, 
            self.data['Start_Time'].iloc[index]
        )
        time_end = self.data['End_Time'].iloc[index]
        embed_text = self.data['Embeddings'].iloc[index]

        label_phase = self._get_label(time_end)
        label_rsd = (self.duration - time_end) / 10000

        return [time_start, embed_text, label_phase, label_rsd]

    def __len__(self):
        return len(self.data)
    
    def get_op_name(self):
        return self.data_path.split('/')[-1][:6]

    def positional_encoding(self, ndim, pos):
        pe = np.zeros((ndim,))
        for t in range(ndim):
            if t % 2 == 0:
                pe[t] = np.sin(pos / np.power(10000, t / ndim))
            elif t % 2 == 1:
                pe[t] = np.cos(pos / np.power(10000, t / ndim))
        return pe

    def _get_duration(self):
        with open(self.annot_path) as f:
            annots = json.load(f)

        file_id = self.data_path.split('/')[-1][:6]
        for annot in annots:
            if annot[0] == file_id:
                return utils.time2sec(annot[3][-1].split('-')[-1])

    def _get_intervals(self):
        phase_dic = {
            'Preparation': 0, 'Puncture': 1, 'GuideWire': 2, 
            'CathPlacement': 3, 'CathPositioning': 4, 
            'CathAdjustment': 5, 'CathControl': 6, 
            'Closing': 7, 'Transition': 8
        }

        with open(self.annot_path) as f:
            annots = json.load(f)

        file_id = self.data_path.split('/')[-1][:6]
        for annot in annots:
            if annot[0] == file_id:
                time_stamps = annot[3]
                phases = annot[4]
                phase_intervals = []
                for t, p in zip(time_stamps, phases):
                    phase_start = max(0, utils.time2sec(t.split("-")[0]) - 1)
                    phase_end = utils.time2sec(t.split("-")[1])
                    phase_intervals.append([phase_dic[p], phase_start, phase_end])
                return phase_intervals

    def _get_label(self, time_e):
        for phase, p_start, p_end in self.phase_intervals:
            if p_start < time_e <= p_end:
                return phase


def get_dataset(annot_path, data_split, model_dim, batch_size):
    datasets = []

    for data_list in data_split:
        split_loaders = [
            DataLoader(
                dataset=OPDataset(data_path, annot_path, model_dim),
                batch_size=batch_size,
                shuffle=False
            ) for data_path in data_list
        ]
        datasets.append(split_loaders)
    
    return datasets
