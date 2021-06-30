import torch
torch.set_num_threads(1)
import warnings
import argparse
from src.data.dataset_blond import TYPE_CLASS
from src.federated.server import Server
from src.federated.client import Client
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


def run(rank, world_size, master_addr):
    if rank == 0:
        config = {
            'batch_size': 128,
            'total_epochs': 30,
            'local_epochs': 1,
            'seq_len': 190,
            'criterion': torch.nn.CrossEntropyLoss(),
            'optim': torch.optim.SGD,
            'optim_kwargs': {'lr': 0.059, 'weight_decay': 0.001},
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
            'model_kwargs': {'name': 'CNN1D', 'num_layers': 4, 'start_size': 28},
            'class_dict': TYPE_CLASS,
            'features': None,
            'experiment_name': None,
            'use_synthetic': False,
            'transfer': True
        }

        feature_dict = {
            'train': [RandomAugment(),
                      MFCC()],
            'val': [RandomAugment(p=0),
                    MFCC()],
        }
        config['features'] = feature_dict

        server = Server(world_size, config)
        server.init_process()
        server.run()
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




