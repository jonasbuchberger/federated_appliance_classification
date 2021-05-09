import os
import torch
from datetime import datetime
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils import ROOT_DIR
from src.models.gan_models import Generator
from src.data.dataset_blond import BLOND


def create_synthetic_data(path_to_data, generator, num_rounds=1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)

    class_dict = {
        'Battery Charger': 0,
        'Daylight': 1,
        'Dev Board': 2,
        'PC': 3,
        'Printer': 4,
        'Projector': 5,
        'Screen Motor': 6,
        'USB Charger': 7
    }
    class_type = {v: k for k, v in class_dict.items()}

    # Columns for new event dataframe
    df = pd.DataFrame(columns=['Medal', 'Socket', 'Appliance', 'Type', 'Timestamp'])

    for medal_id in tqdm(range(1, 15)):

        medal_path = os.path.join(ROOT_DIR, f'data/synthetic_snippets/medal-{medal_id}')
        dataset = BLOND('all', path_to_data, class_dict=class_dict, medal_id=medal_id)
        os.makedirs(medal_path, exist_ok=True)

        for i in range(0, len(dataset) * num_rounds):
            idx = i % len(dataset)
            current, voltage, class_id = dataset[idx]
            noise = (current + current.data.new(current.size()).normal_(0, 0.1)).unsqueeze(0).unsqueeze(0).to(device)
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
    df_real = pd.read_csv(os.path.join(ROOT_DIR, 'data/events_new.csv'), index_col=0)
    df_real['synthetic'] = 0
    df_real = df_real.append(df, ignore_index=True)
    df_real.to_csv(os.path.join(ROOT_DIR, 'data/events_syn.csv'))


if __name__ == '__main__':

    path_to_data = os.path.join(ROOT_DIR, 'data')
    path_to_model = os.path.join(ROOT_DIR, 'notebooks', 'generator_model.pth')

    generator = Generator()
    generator.load_state_dict(torch.load(path_to_model))

    create_synthetic_data(path_to_data, generator, 2)
