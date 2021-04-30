import os
import warnings

from src.data.dataset_blond import get_datalaoders
from src.models.models import BlondConvNet, BlondLstmNet
from src.models.test import test
from src.models.train import train
from src.utils import ROOT_DIR
from src.models.design_of_experiment import lh, to_dict

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


def run_config(path_to_data, **config):
    """

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
    train_loader, val_loader, test_loader = get_datalaoders(path_to_data, config['batch_size'], features=features,
                                                            class_dict=class_dict)

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
    else:
        print(f'Unsupported model: {config["model"]["name"]}')

    trained_model = train(model, train_loader, val_loader, **config)

    model.load_state_dict(torch.load(trained_model))

    test(model, test_loader, **config)


if __name__ == '__main__':
    path_to_data = os.path.join(ROOT_DIR, 'data')

    class_dict = {
        'Laptop': 0,
        'Monitor': 1,
        'USB Charger': 2
    }
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

    features = {
        'train': [RandomAugment(measurement_frequency=6400), ACPower(measurement_frequency=6400)],
        'val': [RandomAugment(measurement_frequency=6400, p=0), ACPower(measurement_frequency=6400)]
    }

    config = {
        'batch_size': 100,
        'num_epochs': 20,
        'seq_len': 197,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optim': torch.optim.SGD,
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 5,
        'model_kwargs': {'name': 'CNN1D', 'num_layers': 2, 'start_size': 15},
        'features': features,
        'class_dict': class_dict,
        'experiment_name': None
    }
    # run_config(path_to_data, **config)

    # lh([lr, weight_decay, num_layers, start_size], num_exp)
    experiments = lh([[0.001, 0.1], [0, 0.001], [1, 4], [10, 30]], 6)
    for i in range(0, experiments.shape[0]):
        config['optim_kwargs']['lr'] = np.round(experiments[i][0], 3)
        config['optim_kwargs']['weight_decay'] = np.round(experiments[i][1], 3)
        config['model_kwargs']['num_layers'] = int(np.round(experiments[i][2], 0))
        config['model_kwargs']['start_size'] = int(np.round(experiments[i][3], 0))
        run_config(path_to_data, **config)
