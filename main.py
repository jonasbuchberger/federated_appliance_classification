import os
import warnings
import numpy as np

from copy import deepcopy
from src.data.dataset_blond import get_datalaoders
from src.models.models import BlondConvNet, BlondLstmNet, BlondResNet
from src.models.test import test
from src.models.train import train
from src.utils import ROOT_DIR
from src.models.design_of_experiment import lh

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


def run_config(path_to_data, **config):
    """ Trains and tests a single model specified with the config.

    Args:
        **config:
    """
    # Calculate input feature dimension for model initialization
    # Get string of feature list
    in_features = 0
    feature_string = '['
    for feature in config['features']['train']:
        in_features += feature.feature_dim
        feature_string += f'{feature.__class__.__name__}_'
    in_features = max(in_features, 1)
    feature_string = feature_string[:-1] + ']'

    # Create experiment name
    if config['experiment_name'] is None:
        config['experiment_name'] = f"{config['model_kwargs']['name']}_" \
                                    f"{config['optim'].__name__}_" \
                                    f"{config['criterion'].__class__.__name__}_" \
                                    f"CLASS_{len(config['class_dict'].keys())}_" \
                                    f"{feature_string}_6400"
        # f"{config['scheduler'].__name__}_"

        config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                             f"wd-{config['optim_kwargs']['weight_decay']}_" \
                             f"nl-{config['model_kwargs']['num_layers']}_" \
                             f"ss-{config['model_kwargs']['start_size']}"

    else:
        config['run_name'] = ''

    # Create datalaoders
    train_loader, val_loader, test_loader = get_datalaoders(path_to_data,
                                                            config['batch_size'],
                                                            use_synthetic=config['use_synthetic'],
                                                            features=config['features'],
                                                            class_dict=config['class_dict'])

    # Initialize model
    if config['model_kwargs']['name'] == 'CNN1D':
        model = BlondConvNet(in_features=in_features,
                             seq_len=config['seq_len'],
                             num_classes=len(config['class_dict'].keys()),
                             num_layers=config['model_kwargs']['num_layers'],
                             out_features=config['model_kwargs']['start_size'])
    elif config['model_kwargs']['name'] == 'LSTM':
        model = BlondLstmNet(in_features=in_features,
                             seq_len=config['seq_len'],
                             num_classes=len(config['class_dict'].keys()),
                             num_layers=config['model_kwargs']['num_layers'],
                             hidden_layer_size=config['model_kwargs']['start_size'])
    elif config['model_kwargs']['name'] == 'RESNET':
        model = BlondResNet(in_features=in_features,
                            seq_len=config['seq_len'],
                            num_classes=len(config['class_dict'].keys()),
                            num_layers=config['model_kwargs']['num_layers'],
                            out_features=config['model_kwargs']['start_size'])
    else:
        print(f'Unsupported model: {config["model"]["name"]}')

    trained_model, best_f1 = train(model, train_loader, val_loader, **config)

    model.load_state_dict(torch.load(trained_model))

    test(model, test_loader, **config)

    return best_f1


def run_experiment(path_to_data, num_experiments=6, **config):
    """ Automatically runs a forward chaining with all features.
        Runs num_experiments experiments sampled by Optimal Latin HyperCube per feature chain.

    Args:
        path_to_data:
        num_experiments:
        **config:
    """
    measurement_frequency = 6400
    # List of all features to test
    feature_list = [ACPower(measurement_frequency=measurement_frequency),
                    COT(measurement_frequency=measurement_frequency),
                    AOT(measurement_frequency=measurement_frequency),
                    DCS(measurement_frequency=measurement_frequency),
                    Spectrogram(measurement_frequency=measurement_frequency),
                    MelSpectrogram(measurement_frequency=measurement_frequency),
                    MFCC(measurement_frequency=measurement_frequency)]

    # Starting dict to build up feature chain
    best_feature_dict = {
        'train': [RandomAugment(measurement_frequency=6400)],
        'val': [RandomAugment(measurement_frequency=6400, p=0)]
    }

    best_feature_f1 = 0
    for _ in range(0, 100):
        run_f1_array = np.zeros(len(feature_list))

        # Runs a hyper parameter search for given feature combinations
        for j, feature in enumerate(feature_list):
            feature_dict_tmp = deepcopy(best_feature_dict)
            feature_dict_tmp['train'].append(feature)
            feature_dict_tmp['val'].append(feature)
            config['features'] = feature_dict_tmp

            # lh([lr, weight_decay, num_layers, start_size], num_exp)
            experiments = lh([[0.001, 0.1], [0, 0.001], [1, 4], [10, 30]], num_experiments)
            best_run_f1 = 0
            for i in range(0, experiments.shape[0]):
                config['optim_kwargs']['lr'] = np.round(experiments[i][0], 3)
                config['optim_kwargs']['weight_decay'] = np.round(experiments[i][1], 3)
                config['model_kwargs']['num_layers'] = int(np.round(experiments[i][2], 0))
                config['model_kwargs']['start_size'] = int(np.round(experiments[i][3], 0))
                f1 = run_config(path_to_data, **config)
                if best_run_f1 < f1:
                    best_run_f1 = f1

            run_f1_array[j] = f1

        best_feature = np.argmax(run_f1_array)
        if best_feature_f1 < np.max(run_f1_array):
            best_feature_f1 = np.max(run_f1_array)
        else:
            # Break when addition of new feature does not increase f1
            break
        # Add the feature that increased f1 by the most to best_feature_dict
        best_feature_dict['train'].append(feature_list[best_feature])
        best_feature_dict['val'].append(feature_list[best_feature])
        feature_list.remove(feature_list[best_feature])


if __name__ == '__main__':
    path_to_data = os.path.join(ROOT_DIR, 'data')

    class_dict = {
        'Laptop': 0,
        'Monitor': 1,
        'USB Charger': 2
    }
    TYPE_CLASS = {
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
        'model_kwargs': {'name': 'LSTM', 'num_layers': 1, 'start_size': 25},
        'class_dict': class_dict,
        'features': None,
        'experiment_name': 'Best_LSTM_with_synthetic',
        'use_synthetic': True
    }

    #run_experiment(path_to_data, **config)

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
                AOT(measurement_frequency=6400)],
    }
    config['features'] = feature_dict
    run_config(path_to_data, **config)


