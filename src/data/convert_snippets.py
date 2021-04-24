import os

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
    df = pd.DataFrame(columns=['Medal', 'Socket', 'Appliance', 'Type', 'Timestamp'])

    for i in range(1, 15):
        files = os.listdir(os.path.join(ROOT_DIR, f'data/event_snippets/medal-{i}'))
        if 'PaxHeader' in files:
            files.remove('PaxHeader')

        for file in files:
            tmp = file.split('_')
            appliance = tmp[0].split('appliance-')[1]
            if 'MacBook Pro 13 Mid-2014' in appliance:
                appliance = "MacBook Pro 13'' Mid-2014"
            timestamp = f'{tmp[1]}_{tmp[2]}_{tmp[3]}_{tmp[4]}'
            medal_id = tmp[5].split('-')[1]
            socket_id = tmp[6].split('-')[1]

            new_row = {'Medal': medal_id, 'Socket': socket_id, 'Appliance': appliance,
                       'Type': APPLIANCE_TYPE[appliance], 'Timestamp': timestamp}
            df = df.append(new_row, ignore_index=True)

            new_file = f'{medal_id}_{socket_id}_{appliance}_{timestamp}.h5'

            if 'appliance' in file:
                file_path = os.path.join(ROOT_DIR, f'data/event_snippets/medal-{i}', file)
                new_file_path = os.path.join(ROOT_DIR, f'data/event_snippets/medal-{i}', new_file)
                os.rename(file_path, new_file_path)
                pass

            df.to_csv(os.path.join(ROOT_DIR, 'data/events_new.csv'))


if __name__ == '__main__':
    convert_dataset()
