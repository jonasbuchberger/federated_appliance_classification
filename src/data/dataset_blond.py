import os

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import Compose
from sklearn.model_selection import KFold

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

    def __init__(self, fold, path_to_data, transform=None, medal_id=None, class_dict=TYPE_CLASS, use_synthetic=False, k_fold=None):
        """

        Args:
            fold (str): 'train, 'val', 'test' or 'all'
            path_to_data (str): Path to the data folder with the events.csv
            transform (torchvision.transform): Transforms to apply on the current wave
            medal_id (int): 1-14 for single medal or None for all
            class_dict (dict): Dict with the desired classes to use for training
            use_synthetic (bool): Use synthetic data for training
        """
        self.transform = transform
        self.path_to_data = path_to_data
        self.fold = fold
        self.class_dict = class_dict
        self.use_synthetic = use_synthetic
        self.k_fold = k_fold

        # Choose labels with or without synthetic data
        if self.use_synthetic:
            self.labels = pd.read_csv(os.path.join(path_to_data, 'events_syn.csv'), index_col=0)
        else:
            self.labels = pd.read_csv(os.path.join(path_to_data, 'events_new.csv'), index_col=0)
            self.labels['synthetic'] = 0

        # Filter labels for classes in class_dict
        self.labels = self.labels[self.labels['Type'].isin(self.class_dict.keys())]

        # Create f-fold set up if desired with 10 folds
        if self.k_fold is not None:
            kf = KFold(n_splits=10, random_state=1000, shuffle=True)
            train_split, test_split = list(kf.split(self.labels))[self.k_fold]
            df_train = self.labels.iloc[train_split]
            df_train['fold'] = 'train'
            df_test = self.labels.iloc[test_split]
            df_test['fold'] = 'test'
            df_val = self.labels.iloc[test_split]
            df_val['fold'] = 'val'
            self.labels = df_train.append((df_val, df_test), ignore_index=True)

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
            self.labels = self.labels[self.labels['Medal'] == medal_id]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        folder = 'synthetic' if row['synthetic'] else 'event'
        path = os.path.join(self.path_to_data, f'{folder}_snippets/medal-{row["Medal"]}')
        file_name = os.path.join(f'{row["Medal"]}_{row["Socket"]}_{row["Appliance"]}_{row["Timestamp"]}.h5')
        f = h5py.File(os.path.join(path, file_name), 'r')

        # Cut length of measurement window
        current = torch.as_tensor(f['data']['block0_values'][:, 1])
        voltage = torch.as_tensor(f['data']['block0_values'][:, 0])

        if not row['synthetic']:
            # Shifts event window to start with a new cycle
            tmp = (current - torch.mean(current)) / torch.std(current)
            idx = torch.where(torch.diff(torch.signbit(tmp[:250])))[0][0]

            current = current[idx: 24576 + idx]
            voltage = voltage[idx: 24576 + idx]

        # Apply feature transform on current/voltage, if no transform applied return (current, voltage, class)
        class_num = int(row['Class'])
        sample = (current, voltage, None, class_num)
        if self.transform:
            _, _, features, _ = self.transform(sample)
            return features.float(), class_num
        else:
            return sample[0].float(), sample[1].float(), sample[3]


def get_datalaoders(path_to_data, batch_size, medal_id=None, use_synthetic=False, features=None, class_dict=TYPE_CLASS, k_fold=None):
    """ Returns data loaders

    Args:
        path_to_data (string): Path to the dataset
        batch_size (int): Size of batches
        medal_id (int): 1-14 for single medal or None for all
        features (dict): Dict containing the train and val/test features
        use_synthetic (bool): Use synthetic data for training
        class_dict (dict): Dict of type class mapping

    Returns:
        train_loader (torch.utils.data.DataLoader)
        val_loader (torch.utils.data.DataLoader)
        test_loader (torch.utils.data.DataLoader)
    """
    num_workers = 2

    train_set = BLOND(path_to_data=path_to_data,
                      fold='train',
                      transform=Compose(features['train']) if features is not None else features,
                      medal_id=medal_id,
                      use_synthetic=use_synthetic,
                      class_dict=class_dict,
                      k_fold=k_fold)
    val_set = BLOND(path_to_data=path_to_data,
                    fold='val',
                    transform=Compose(features['val']) if features is not None else features,
                    medal_id=medal_id,
                    use_synthetic=use_synthetic,
                    class_dict=class_dict,
                    k_fold=k_fold)
    test_set = BLOND(path_to_data=path_to_data,
                     fold='test',
                     transform=Compose(features['val']) if features is not None else features,
                     medal_id=medal_id,
                     use_synthetic=use_synthetic,
                     class_dict=class_dict,
                     k_fold=k_fold)

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
    BLOND('train', path, k_fold=1)

    t, _, _ = get_datalaoders(path, 10, class_dict=class_dict)

    for sample in t:
        _, c = sample
        print(c)
