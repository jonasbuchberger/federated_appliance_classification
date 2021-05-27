import os
import warnings
import argparse
from src.data.dataset_blond import TYPE_CLASS
from src.federated.server import Server
from src.federated.client import Client

import torch
torch.set_num_threads(1)

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("rank", help="Rank of current worker.", type=int)
    parser.add_argument("world_size", help="Total size of worker.", type=int)
    parser.add_argument("--master_addr", help="IP of master.", type=str, required=False, default='127.0.0.1')

    args = parser.parse_args()
    world_size = args.world_size
    rank = args.rank
    master_addr = args.master_addr

    if rank == 0:
        config = {
            'batch_size': 10,
            'total_epochs': 2,
            'local_epochs': 1,
            'seq_len': 190,
            'criterion': torch.nn.CrossEntropyLoss(),
            'optim': torch.optim.SGD,
            'optim_kwargs': {'lr': 0.059, 'weight_decay': 0.0},
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
            'model_kwargs': {'name': 'CNN1D', 'num_layers': 4, 'start_size': 28},
            'class_dict': TYPE_CLASS,
            'features': None,
            'experiment_name': None,
            'use_synthetic': True,
        }

        feature_dict = {
            'train': [RandomAugment(measurement_frequency=6400),
                      MFCC(measurement_frequency=6400)],
            'val': [RandomAugment(measurement_frequency=6400, p=0),
                    MFCC(measurement_frequency=6400)]
        }
        config['features'] = feature_dict

        server = Server(world_size, config)
        server.init_process()
        server.run()

    else:
        client = Client(rank, world_size, master_addr=master_addr)
        client.init_process()
        client.run()

