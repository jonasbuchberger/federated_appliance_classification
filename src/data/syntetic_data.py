import os
import torch
from datetime import datetime
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn
from src.utils import ROOT_DIR


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(DenseBlock, self).__init__()
        if activation == 'sig':
            self.activation = nn.Sigmoid()
        if activation == 'softmax':
            self.activation = nn.Softmax()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.bn = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn = self.bn(x)
        conv1 = self.activation(self.conv1(bn))
        conv2 = self.activation(self.conv2(conv1))

        c2_dense = self.activation(torch.cat([conv1, conv2], 1))
        conv3 = self.activation(self.conv3(c2_dense))
        c3_dense = self.activation(torch.cat([conv1, conv2, conv3], 1))
        conv4 = self.activation(self.conv4(c3_dense))
        c4_dense = self.activation(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.activation(self.conv5(c4_dense))
        c5_dense = self.activation(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=7, stride=3, padding=0),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(20),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(30),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(40),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            DenseBlock(50, 1, 'relu')
        )
        self.linear = nn.Linear(903, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=7, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(10),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(10, 20, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(20),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(20, 30, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(30),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(40),
            # nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),

            nn.ConvTranspose1d(40, 30, kernel_size=7, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.ConvTranspose1d(30, 20, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.ConvTranspose1d(20, 10, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.ConvTranspose1d(10, 5, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(5),
            # -nn.ReLU(),
            nn.ConvTranspose1d(5, 1, kernel_size=3, stride=1, padding=0, bias=False),
            # -nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        # x = x.reshape(x.size(0), -1)
        # print(x.shape)
        return x


class BLOND(Dataset):

    def __init__(self, path_to_data, class_dict, medal_id=None):
        """

        Args:
            path_to_data (str): Path to the data folder with the events.csv
            class_dict (dict): Dict with the desired classes to use for training
            medal_id (int): 1-14 for single medal or None for all
        """

        self.path_to_data = path_to_data
        self.class_dict = class_dict

        self.labels = pd.read_csv(os.path.join(path_to_data, 'events_medal.csv'), index_col=0)

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

        # Filter for medal unit
        if medal_id is not None:
            self.labels = self.labels[self.labels['Medal'] == str(medal_id)]

    def __len__(self):
        return max(len(self.labels), 128)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        path = os.path.join(self.path_to_data, f'event_snippets/medal-{row["Medal"]}')
        file_name = os.path.join(f'{row["Medal"]}_{row["Socket"]}_{row["Appliance"]}_{row["Timestamp"]}.h5')
        f = h5py.File(os.path.join(path, file_name), 'r')

        # Cut length of measurement window
        current = torch.as_tensor(f['data']['block0_values'][:, 1])
        voltage = torch.as_tensor(f['data']['block0_values'][:, 0])

        current = current[: 25598]
        voltage = voltage[: 25598]

        # Apply feature transform on current/voltage, if no transform applied return (current, voltage, class)
        class_num = int(row['Class'])

        return current.float(), voltage.float(), class_num


def create_synthetic_data(path_to_data, generator, class_dict, num_rounds=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)

    class_type = {v: k for k, v in class_dict.items()}

    # Columns for new event dataframe
    df = pd.DataFrame(columns=['Medal', 'Socket', 'Appliance', 'Type', 'Timestamp'])

    for medal_id in tqdm(range(1, 15)):

        medal_path = os.path.join(ROOT_DIR, f'data/synthetic_snippets/medal-{medal_id}')
        dataset = BLOND(path_to_data, class_dict, medal_id)
        os.makedirs(medal_path, exist_ok=True)

        for i in range(0, len(dataset) * num_rounds):
            idx = i % len(dataset)
            current, voltage, class_id = dataset[idx]
            noise = (current + current.data.new(current.size()).normal_(0, 0.01)).unsqueeze(0).unsqueeze(0).to(device)
            # Generate new sample with original data and added noise
            gen_current = generator(noise).detach().cpu().squeeze()

            date = datetime.now().date()
            time = datetime.now().time()
            timestamp = f'{date} {time}'.replace(':', '_')
            # Append new row to dataframe
            new_row = {'Medal': medal_id, 'Socket': 0, 'Appliance': class_type[class_id],
                       'Type': class_type[class_id], 'Timestamp': timestamp}
            df = df.append(new_row, ignore_index=True)

            file = f'{medal_id}_{0}_{class_type[class_id]}_{timestamp}.h5'
            file_path = os.path.join(medal_path, file)

            # Safe synthetic sample in h5 store
            f = h5py.File(file_path, "w")
            f.create_dataset('data/block0_values', data=(np.stack((voltage, gen_current), axis=1)))
            f.close()

    # Append labels of real and synthetic data
    df['fold'] = 'train'
    df['synthetic'] = 1
    if os.path.isfile(os.path.join(ROOT_DIR, 'data/events_syn.csv')):
        df_2 = pd.read_csv(os.path.join(ROOT_DIR, 'data/events_syn.csv'), index_col=0)
    else:
        df_2 = pd.read_csv(os.path.join(ROOT_DIR, 'data/events_medal.csv'), index_col=0)
        df_2['synthetic'] = 0
    df_2 = df_2.append(df, ignore_index=True)
    df_2.to_csv(os.path.join(ROOT_DIR, 'data/events_syn.csv'))


if __name__ == '__main__':
    path_to_data = os.path.join(ROOT_DIR, 'data')

    class_dict = {
        #'Battery Charger': 0,
        'Daylight': 1,
        #'Dev Board': 2,
        #'Fan': 3,
        #'Kettle': 4,
        #'Laptop': 5,
        #'Monitor': 6,
        #'PC': 7,
        #'Printer': 8,
        #'Projector': 9,
        #'Screen Motor': 10,
        #'USB Charger': 11,
        }

    for class_i in list(class_dict.keys()):
        path_to_model = os.path.join(ROOT_DIR, 'notebooks', 'generator', f'generator_model_{class_i}.pth')
        generator = Generator()
        generator.load_state_dict(torch.load(path_to_model))
        tmp_dict = {f'{class_i}': class_dict[class_i]}
        create_synthetic_data(path_to_data, generator, tmp_dict, 6)
