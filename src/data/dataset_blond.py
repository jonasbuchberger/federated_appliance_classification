import os

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import Compose

from src.utils import ROOT_DIR

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

    def __init__(self, fold, path_to_data, transform=None, medal_id=None, class_dict=TYPE_CLASS):
        """

        Args:
            fold (str): 'train, 'val', 'test' or 'all'
            path_to_data (str): Path to the data folder with the events.csv
            transform (torchvision.transform): Transforms to apply on the current wave
            medal_id (int): 1-14 for single medal or None for all
            class_dict (dict): Dict with the desired classes to use for training
        """
        self.transform = transform
        self.path_to_data = path_to_data
        self.fold = fold
        self.class_dict = class_dict

        self.labels = pd.read_csv(os.path.join(path_to_data, 'events_new.csv'), index_col=0)
        # Filter labels for classes in class_dict
        self.labels = self.labels[self.labels['Type'].isin(self.class_dict.keys())]

        # Calculate class weights
        self.class_weights = len(self.labels) / self.labels['Type'].value_counts()

        classes = torch.zeros(len(self.labels))
        sample_weights = torch.zeros(len(self.labels))
        # Add sample weights and class identifiers to labels dataframe
        for i, type_i in enumerate(self.class_weights.index):
            idx = (self.labels['Type'] == type_i).values
            classes[idx] = self.class_dict[type_i]
            sample_weights[idx] = self.class_weights[type_i]
        self.labels['Class'] = classes
        self.labels['Weight'] = sample_weights

        # Fold dataset
        if self.fold != 'all':
            self.labels = self.labels[self.labels['fold'] == fold]

        # Filter for medal unit
        if medal_id is not None:
            self.labels = self.labels[self.labels['Medal_Nr'] == medal_id]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        path = os.path.join(self.path_to_data, f'event_snippets/medal-{row["Medal"]}')
        file_name = os.path.join(f'{row["Medal"]}_{row["Socket"]}_{row["Appliance"]}_{row["Timestamp"]}.h5')
        f = h5py.File(os.path.join(path, file_name), 'r')

        # Cut length of measurement window
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

        # Apply feature transform on current/voltage, if no transform applied return (current, voltage, None, class)
        class_num = int(row['Class'])

        sample = (current, voltage, None, class_num)
        if self.transform:
            _, _, features, _ = self.transform(sample)
            return features.float(), class_num
        else:
            return sample


def get_datalaoders(path_to_data, batch_size, medal_id=None, features=None, class_dict=TYPE_CLASS):
    """ Returns data loaders

    Args:
        path_to_data (string): Path to the dataset
        batch_size (int): Size of batches
        medal_id (int): 1-14 for single medal or None for all
        features (dict): Dict containing the train and val/test features
        class_dict (dict): Dict of type class mapping

    Returns:
        train_loader (torch.utils.data.DataLoader)
        val_loader (torch.utils.data.DataLoader)
        test_loader (torch.utils.data.DataLoader)
    """
    num_workers = 2
    if features is None:
        features = {'train': None, 'val': None, 'test': None}

    train_set = BLOND(path_to_data=path_to_data,
                      fold='train',
                      transform=Compose(features['train']),
                      medal_id=medal_id,
                      class_dict=class_dict)
    val_set = BLOND(path_to_data=path_to_data,
                    fold='val',
                    transform=Compose(features['val']),
                    medal_id=medal_id,
                    class_dict=class_dict)
    test_set = BLOND(path_to_data=path_to_data,
                     fold='test',
                     transform=Compose(features['val']),
                     medal_id=medal_id,
                     class_dict=class_dict)

    sampler = WeightedRandomSampler(train_set.labels['Weight'], len(train_set), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    class_dict = {
        'Laptop': 0,
        'Monitor': 1,
        'USB Charger': 2
    }

    path = os.path.join(ROOT_DIR, 'data')
    # print(len(BLOND('val', path)))

    t, _, _ = get_datalaoders(path, 10, class_dict=class_dict)

    for sample in t:
        _, c = sample
        print(c)
