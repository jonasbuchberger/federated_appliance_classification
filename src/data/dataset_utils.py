import os

import numpy as np
import pandas as pd

from src.utils import ROOT_DIR

APPLIANCE_TYPE = {
    'Akura': 'USB Charger',
    'Apple MD836ZM': 'USB Charger',
    'Dell Optiplex 7040': 'PC',
    'Dell P2210': 'Monitor',
    'Dell T3600': 'PC',
    'Dell U2711': 'Monitor',
    'Dell U2713Hb': 'Monitor',
    'Dell UP2716D': 'Monitor',
    'Dell XPS13': 'Laptop',
    'Epson EB-65950': 'Projector',
    'FPGA Xilinx ML505': 'Dev Board',
    'Fujitsu-Siemens P17-1': 'Monitor',
    'HP Laserjet Pro 400': 'Printer',
    'Hama 0091321': 'USB Charger',
    'Kraftmax BC4000 Pro': 'Battery Charger',
    'Lenovo Carbon X1': 'Laptop',
    'Lenovo L540': 'Laptop',
    'Lenovo T420': 'Laptop',
    'Lenovo T450': 'Laptop',
    'Lenovo X230 i5': 'Laptop',
    'Lenovo X230 i7': 'Laptop',
    "MacBook Pro 13'' Mid-2014": 'Laptop',
    "MacBook Pro 15'' Mid-2014": 'Laptop',
    'Philips HF3430': 'Daylight',
    'Projecta DC 485': 'Screen Motor',
    'Samsung Travel': 'USB Charger',
    'Schenker W502': 'Laptop',
    'Sony Vaio VGN FW54M': 'Laptop',
    'generic': 'USB Charger',
    'inateck UC2001': 'USB Charger'
}


def convert_dataset():
    """ Used to rename all event files and create events.csv

    """
    # Columns for new event dataframe
    df = pd.DataFrame(columns=['Medal', 'Socket', 'Appliance', 'Type', 'Timestamp'])

    for i in range(1, 15):
        files = os.listdir(os.path.join(ROOT_DIR, f'data/event_snippets/medal-{i}'))
        if 'PaxHeader' in files:
            files.remove('PaxHeader')

        for file in files:
            tmp = file.split('_')
            appliance = tmp[0].split('appliance-')[1]
            # Needed because events labeled with two different versions
            if 'MacBook Pro 13 Mid-2014' in appliance:
                appliance = "MacBook Pro 13'' Mid-2014"

            # Recreate event information
            timestamp = f'{tmp[1]}_{tmp[2]}_{tmp[3]}_{tmp[4]}'
            medal_id = tmp[5].split('-')[1]
            socket_id = tmp[6].split('-')[1]

            # Append new row to dataframe
            new_row = {'Medal': medal_id, 'Socket': socket_id, 'Appliance': appliance,
                       'Type': APPLIANCE_TYPE[appliance], 'Timestamp': timestamp}
            df = df.append(new_row, ignore_index=True)

            new_file = f'{medal_id}_{socket_id}_{appliance}_{timestamp}.h5'
            file_path = os.path.join(ROOT_DIR, f'data/event_snippets/medal-{i}', file)
            new_file_path = os.path.join(ROOT_DIR, f'data/event_snippets/medal-{i}', new_file)
            # Renames given file with new file name
            os.rename(file_path, new_file_path)

            df.to_csv(os.path.join(ROOT_DIR, 'data/events_new.csv'))


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
