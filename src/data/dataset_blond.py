import h5py
import os
import pandas as pd
import torch
from src.utils import ROOT_DIR
from torch.utils.data import Dataset

TYPE_CLASS = {
    'Battery Charger': 0,
    'Daylight': 1,
    'Dev Board': 2,
    'Laptop': 3,
    'Monitor': 4,
    'PC': 5,
    'Printer': 6,
    'Projector': 7,
    'Screen Motor': 8,
    'USB Charger': 9
}

class BLOND(Dataset):

    def __init__(self, fold, path_to_data, transform=None, medal_id=None):

        self.transform = transform
        self.path_to_data = path_to_data
        self.fold = fold

        self.labels = pd.read_csv(os.path.join(path_to_data, 'events_new.csv'), index_col=0)

        if medal_id is not None:
            self.labels = self.labels[self.labels['Medal_Nr'] == medal_id]

        self.labels_dict = {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        path = os.path.join(self.path_to_data, f'event_snippets/medal-{row["Medal"]}')
        file_name = os.path.join(f'{row["Medal"]}_{row["Socket"]}_{row["Appliance"]}_{row["Timestamp"]}.h5')
        f = h5py.File(os.path.join(path, file_name), 'r')

        current = torch.as_tensor(f['data']['block0_values'][:, 1])[300:][:25000]
        voltage = torch.as_tensor(f['data']['block0_values'][:, 0])[300:][:25000]
        """
        # Shifts event widnow to start with a new cycle
        if str(events['Phase'].loc[i]) == 'A':
            idx = np.where(np.diff(np.signbit(window['Current A'][:250])))[0][0]
        else:
            idx = np.where(np.diff(np.signbit(window['Current B'][:250])))[0][0]

        window = data[lower + idx: upper + idx]
        """

        sample = (current, voltage, None, TYPE_CLASS[row['Type']])
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    path = os.path.join(ROOT_DIR, 'data')
    x = BLOND('train', path)[0]
