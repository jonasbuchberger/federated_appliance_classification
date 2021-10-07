import os
import time
import torch
import socket

torch.set_num_threads(1)
import warnings
import argparse
from src.data.dataset_blond import TYPE_CLASS
from src.federated.server import Server
from src.federated.client import Client
from src.federated.federated_utils import get_pi_usage
from datetime import datetime, timedelta
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


def run(rank, world_size, master_addr):
    if rank == 0:
        config = {
            'setting': 'noniid',
            'batch_size': 128,
            'epochs': {'agg_rounds': 100, 'local_steps': 4, 'mode': 'step'},
            'logging_factor': 1,
            'seq_len': 190,
            'criterion': torch.nn.CrossEntropyLoss(),
            'optim': torch.optim.SGD,
            'optim_kwargs': {'lr': 0.045, 'weight_decay': 0.001},
            'model_kwargs': {'name': 'LSTM', 'num_layers': 1, 'start_size': 23},
            # 'optim_kwargs': {'lr': 0.055, 'weight_decay': 0.0},
            # 'model_kwargs': {'name': 'CNN1D', 'num_layers': 4, 'start_size': 19},
            # 'optim_kwargs': {'lr': 0.075, 'weight_decay': 0.001},
            # 'model_kwargs': {'name': 'DENSE', 'num_layers': 3, 'start_size': 12},
            # 'optim_kwargs': {'lr': 0.052, 'weight_decay': 0.001},
            # 'model_kwargs': {'name': 'RESNET', 'num_layers': 4, 'start_size': 20},
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
            'scheduler_kwargs': {'T_max': 100},
            'class_dict': TYPE_CLASS,
            'features': None,
            'experiment_name': None,
            'use_synthetic': True,
            'transfer': False,
            'transfer_kwargs': {'lr': 0.075, 'weight_decay': 0.0, 'num_epochs': 10},
            'local_test': False,
            'weighted': False,
            'early_stopping': 100,
        }

        feature_dict = {
            'train': [RandomAugment(),
                      ACPower(),
                      MFCC()],
            'val': [RandomAugment(p=0),
                    ACPower(),
                    MFCC()],
        }
        config['features'] = feature_dict

        start_time = datetime.now()
        start_time = start_time - timedelta(seconds=30)
        server = Server(world_size, config)
        server.init_process()
        log_path = server.run()
        host = socket.gethostname()

        if 'raspi' in host:
            time.sleep(30)
            end_time = datetime.now()
            print('Retrieving Pi usage.')
            get_pi_usage(start_time, end_time, os.path.join(log_path, 'pi_logs'))

        print('-----------------Finished-----------------')
    else:
        client = Client(rank, world_size, master_addr=master_addr)
        client.init_process()
        client.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("world_size", help="Total size of workers.", type=int)
    parser.add_argument("-r", "--rank", help="Rank of current worker.", type=int, required=False, default=None)
    parser.add_argument("-m", "--master_addr", help="IP of master.", type=str, required=False, default='127.0.0.1')

    args = parser.parse_args()
    world_size = args.world_size
    rank = args.rank
    master_addr = args.master_addr

    if rank is None:
        processes = []
        mp.set_start_method("spawn")
        for rank in range(0, world_size):
            p = mp.Process(target=run, args=(rank, world_size, master_addr))
            print(f'Started worker {rank}.')
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        run(rank, world_size, master_addr)
