import os

import numpy as np
import pandas as pd

from src.utils import ROOT_DIR


def create_dataset_splits(split=0.1):
    """ Splits labels into train, val, test while label distribution is kept

    Args:
        split (float): Size of test and val split (0.1 = 80/10/10)
    """
    path = os.path.join(ROOT_DIR, 'data/events_new.csv')
    labels = pd.read_csv(path, index_col=0)
    labels['fold'] = 'train'
    # Get the number of samples per class
    class_count = labels['Type'].value_counts()

    for type_i in labels['Type'].unique():
        if pd.notnull(type_i):
            # Get all indices of a class of type_i
            class_indices = labels[labels['Type'] == type_i].index
            # Draw #(class_count * 2 * split) random samples for test and validation
            rnd_indices = np.random.choice(class_indices, size=int(class_count[type_i] * 2 * split), replace=False)
            # Split indices into test and val
            rnd_indices_val, rnd_indices_test = np.array_split(rnd_indices, 2)
            # Assign correct labels for indices
            labels.loc[rnd_indices_val, 'fold'] = 'val'
            labels.loc[rnd_indices_test, 'fold'] = 'test'

    labels.to_csv(path)


if __name__ == '__main__':
    # convert_dataset()
    create_dataset_splits()
