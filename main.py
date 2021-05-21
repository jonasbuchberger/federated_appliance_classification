import os
import warnings
from src.utils import ROOT_DIR
from src.models.experiment_utils import run_experiment, run_k_fold, run_config

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *

if __name__ == '__main__':
    path_to_data = os.path.join(ROOT_DIR, 'data')

    """
    class_dict = {
        'Battery Charger': 0,
        'Daylight': 1,
        'Dev Board': 2,
        'Laptop': 3,
        'Monitor': 4,
        'PC': 5,
        'Printer': 6,
        'Projector': 7,
        'Screen Motor': 8,
        'USB Charger': 9
    }
    """
    class_dict = {
        'Dev Board': 0,
        'Laptop': 1,
        'Monitor': 2,
        'PC': 3,
        'Printer': 4,
        'Projector': 5,
        'Screen Motor': 6,
        'USB Charger': 7
    }


    config = {
        'batch_size': 100,
        'num_epochs': 20,
        'seq_len': 190,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optim': torch.optim.SGD,
        'optim_kwargs': {'lr': 0.059, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 5,
        'model_kwargs': {'name': 'LSTM', 'num_layers': 1, 'start_size': 28},
        'class_dict': class_dict,
        'features': None,
        'experiment_name': None,
        'use_synthetic': True,
    }

    for m in ['CNN1D', 'LSTM', 'RESNET']:
        config['model_kwargs']['name'] = m
        run_experiment(path_to_data, **config)

    feature_dict = {
        'train': [RandomAugment(measurement_frequency=6400),
                  ACPower(measurement_frequency=6400),
                  Spectrogram(measurement_frequency=6400),
                  COT(measurement_frequency=6400),
                  AOT(measurement_frequency=6400)],
        'val': [RandomAugment(measurement_frequency=6400, p=0),
                ACPower(measurement_frequency=6400),
                Spectrogram(measurement_frequency=6400),
                COT(measurement_frequency=6400),
                AOT(measurement_frequency=6400)]
    }
    config['features'] = feature_dict
    #run_config(path_to_data, **config)
    #run_k_fold(path_to_data, **config)



