import os
import warnings
import argparse
import torch.distributed as dist
from src.data.dataset_blond import TYPE_CLASS
from src.federated.server import run_server
from src.federated.client import run_client

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


def init_process(rank, size, backend='gloo', master_addr='127.0.0.1', master_port='29500'):
    """ Initializes the federated learning setup

    Args:
        rank (int): Worker identifier, 0: server
        size (int): World size equals #clients + server
        backend (string): 'gloo' or 'nccl'
        master_addr (string): Ip address of server
        master_port (string): Port of the server
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    init_method = None
    if 'tcp' in master_addr:
        init_method = f'{master_addr}:{master_port}'
    dist.init_process_group(backend, rank=rank, world_size=size, init_method=init_method)

    # Rank 0 is set to aggregation server
    if rank != 0:
        run_client(rank, size)
    else:
        config = {
            'batch_size': 10,
            'total_epochs': 36,
            'local_epochs': 6,
            'seq_len': 190,
            'criterion': torch.nn.CrossEntropyLoss(),
            'optim': torch.optim.SGD,
            'optim_kwargs': {'lr': 0.059, 'weight_decay': 0.0},
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
            'model_kwargs': {'name': 'CNN1D', 'num_layers': 4, 'start_size': 22},
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

        run_server(size, config)


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", help="Rank of current worker.", type=int)
    parser.add_argument("world_size", help="Total size of worker.", type=int)

    args = parser.parse_args()
    world_size = args.world_size
    rank = args.rank

    init_process(rank, world_size)
    """
    import torch.multiprocessing as mp
    size = 5
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size))
        print(f'Started worker {rank}.')
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
