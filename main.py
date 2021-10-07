import torch
torch.set_num_threads(2)

import os
import warnings
import time
import socket
from datetime import datetime, timedelta
from src.utils import ROOT_DIR
from src.models.experiment_utils import run_experiment, run_k_fold, run_config

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *
from src.data.dataset_blond import TYPE_CLASS
from src.federated.federated_utils import get_pi_usage

if __name__ == '__main__':

    path_to_data = os.path.join(ROOT_DIR, 'data')

    config = {
        'batch_size': 128,
        'num_epochs': 20,
        'seq_len': 190,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optim': torch.optim.SGD,
        'optim_kwargs': {'lr': 0.052, 'weight_decay': 0.001},
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
        'scheduler_kwargs': {'T_max': 50},
        # 'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'model_kwargs': {'name': 'RESNET', 'num_layers': 4, 'start_size': 20},
        'early_stopping': 50,
        'class_dict': TYPE_CLASS,
        'features': None,
        'experiment_name': None,
        'use_synthetic': True,
    }

    # Runs complete hyperparameter, architecture and feature search
    # for m in ['CNN1D', 'LSTM', 'RESNET', 'DENSE']:
    #    config['model_kwargs']['name'] = m
    #    run_experiment(path_to_data, **config)

    feature_dict = {
        'train': [RandomAugment(),
                  MFCC(),
                  COT()],
        'val': [RandomAugment(p=0),
                MFCC(),
                COT()]
    }
    config['features'] = feature_dict

    # Runs k-fold experiment with given config
    # run_k_fold(path_to_data, 10, **config)

    # Runs single run with given config and logs performance on pis
    start_time = datetime.now()
    start_time = start_time - timedelta(seconds=30)

    _, log_path = run_config(path_to_data, **config)

    host = socket.gethostname()
    if 'raspi' in host:
        time.sleep(30)
        end_time = datetime.now()
        print('Retrieving Pi usage.')
        get_pi_usage(start_time, end_time, os.path.join(log_path, 'pi_logs'), pis=[host[-2:]])





