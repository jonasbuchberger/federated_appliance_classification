import torch
torch.set_num_threads(2)

import os
import warnings
from src.utils import ROOT_DIR
from src.models.experiment_utils import run_experiment, run_k_fold, run_config

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *

if __name__ == '__main__':

    path_to_data = os.path.join(ROOT_DIR, 'data')

    class_dict = = {
        'Battery Charger': 0,
        'Daylight': 1,
        'Dev Board': 2,
        'Fan': 3,
        'Kettle': 4,
        'Laptop': 5,
        'Monitor': 6,
        'PC': 7,
        'Printer': 8,
        'Projector': 9,
        'Screen Motor': 10,
        'USB Charger': 11,
    }

    config = {
        'batch_size': 100,
        'num_epochs': 20,
        'seq_len': 190,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optim': torch.optim.SGD,
        'optim_kwargs': {'lr': 0.026, 'weight_decay': 0.001},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 5,
        'model_kwargs': {'name': 'CNN1D', 'num_layers': 4, 'start_size': 18},
        'class_dict': class_dict,
        'features': None,
        'experiment_name': None,
        'use_synthetic': False,
    }

    for m in ['CNN1D', 'LSTM', 'RESNET', 'DENSE']:
        config['model_kwargs']['name'] = m
        run_experiment(path_to_data, **config)

    feature_dict = {
        'train': [RandomAugment(),
                  MFCC()],
        'val': [RandomAugment(p=0),
                MFCC()]
    }

    # Model per medal
    #for medal_id in range(1, 16):
    #    config['features'] = feature_dict
    #    config['experiment_name'] = f'medal_{medal_id}'
    #    config['medal_id'] = medal_id
    #    run_config(path_to_data, **config)

    #run_config(path_to_data, **config)
    #run_k_fold(path_to_data, 10, **config)



