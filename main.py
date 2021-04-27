import os
import warnings

from src.data.dataset_blond import get_datalaoders
from src.models.models import BlondConvNet, BlondLstmNet
from src.models.test import test
from src.models.train import train
from src.utils import ROOT_DIR

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *

if __name__ == '__main__':

    path_to_data = os.path.join(ROOT_DIR, 'data')
    experiment_name = None
    seq_len = 23

    class_dict = {
        'Laptop': 0,
        'Monitor': 1,
        'USB Charger': 2
    }

    features = {
        'train': [RandomAugment(), ACPower()],
        'val': [RandomAugment(p=0), ACPower()]
    }

    config = {
        'batch_size': 60,
        'num_epochs': 1,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optim': torch.optim.SGD,
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 5,
        'model_kwargs': {'name': 'CNN1D', 'num_layers': 2, 'hidden_layer': 15},
        'features': features,
        'class_dict': class_dict
    }

    # Calculate input feature dimension for model initialization
    # Get string of feature list
    in_features = 0
    feature_string = '['
    for feature in features['train']:
        in_features += feature.feature_dim
        feature_string += f'{feature.__class__.__name__}_'
    in_features = max(in_features, 1)
    feature_string = feature_string[:-1] + ']'

    # Create experiment name
    if experiment_name is None:
        config['experiment_name'] = f"{config['model_kwargs']['name']}_" \
                                    f"{config['optim'].__name__}_" \
                                    f"{config['criterion'].__class__.__name__}_" \
                                    f"CLASS_{len(class_dict.keys())}"
                                    # f"{config['scheduler'].__name__}_"

        config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                             f"wd-{config['optim_kwargs']['weight_decay']}_" \
                             f"nl-{config['model_kwargs']['num_layers']}_" \
                             f"nh-{config['model_kwargs']['hidden_layer']}_" \
                             f"{feature_string}"
    else:
        config['experiment_name'] = experiment_name
        config['run_name'] = ''

    # Create datalaoders
    train_loader, val_loader, test_loader = get_datalaoders(path_to_data, config['batch_size'], features=features,
                                                            class_dict=class_dict)

    # Initialize model
    if config['model_kwargs']['name'] == 'CNN1D':
        model = BlondConvNet(in_features=in_features,
                             seq_len=seq_len,
                             num_classes=len(class_dict.keys()),
                             num_layers=config['model_kwargs']['num_layers'],
                             hidden_layer_size=config['model_kwargs']['hidden_layer'])
    elif config['model_kwargs']['name'] == 'LSTM':
        model = BlondLstmNet(in_features=in_features,
                             seq_len=seq_len,
                             num_classes=len(class_dict.keys()),
                             batch_size=config['batch_size'],
                             num_layers=config['model_kwargs']['num_layers'],
                             hidden_layer_size=config['model_kwargs']['hidden_layer'])
    else:
        print(f'Unsupported model: {config["model"]["name"]}')

    trained_model = train(model, train_loader, val_loader, **config)

    model.load_state_dict(torch.load(trained_model))

    test(model, test_loader, **config)
