import torch
import torchvision
from src.utils import ROOT_DIR
from torch.utils.data import Dataset

class BLOND(Dataset):

    def __init__(self, path_to_data, fold, transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        """"
        # Shifts event widnow to start with a new cycle
        if str(events['Phase'].loc[i]) == 'A':
            idx = np.where(np.diff(np.signbit(window['Current A'][:250])))[0][0]
        else:
            idx = np.where(np.diff(np.signbit(window['Current B'][:250])))[0][0]

        window = data[lower + idx: upper + idx]
        """
        pass