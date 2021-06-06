import warnings
import numpy as np

from datetime import datetime
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import Compose
from src.data.dataset_blond import BLOND, TYPE_CLASS
from src.models.models import BlondConvNet, BlondLstmNet, BlondResNet
from src.models.test import test
from src.models.train import train
from src.models.design_of_experiment import lh

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *


def run_config(path_to_data, **config):
    """ Trains and tests a single model specified with the config.

    Args:
        path_to_data (string): Path to the data directory
        **config: batch_size (int): Size of the batches
                  num_epochs (int): Max number of train epochs
                  seq_len (int): Length of model input time series after processing
                  criterion (torch.nn): Loss function
                  optim (torch.optim): Optimizer
                  optim_kwargs (dict): Optimizer kwargs
                  scheduler (torch.optim): LR scheduler for training
                  scheduler_kwargs (dict): Scheduler kwargs
                  early_stopping (int): Number of epchs or None for no early stopping
                  model_kwargs (dict): Model initialization parameters
                  class_dict (dict): Dictionary of desired classes
                  features (dict): Dictionary of val and train features as list
                  experiment_name (string): Name to store the experiment results or None
                  use_synthetic (bool): Use synthetic data for training

    Returns:
        (float): Best F1 score on validation set
    """

    # Get string of feature list
    feature_string = '['
    for feature in config['features']['train']:
        feature_string += f'{feature.__class__.__name__}_'
    feature_string = feature_string[:-1] + ']'

    # Create experiment name
    if config['experiment_name'] is None:
        config['experiment_name'] = f"{config['model_kwargs']['name']}_" \
                                    f"{config['optim'].__name__}_" \
                                    f"{config['criterion'].__class__.__name__}_" \
                                    f"CLASS_{len(config['class_dict'].keys())}_" \
                                    f"{feature_string}_6400_Synthetic"
        # f"{config['scheduler'].__name__}_"

        config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                             f"wd-{config['optim_kwargs']['weight_decay']}_" \
                             f"nl-{config['model_kwargs']['num_layers']}_" \
                             f"ss-{config['model_kwargs']['start_size']}"

    else:
        config['run_name'] = config.get('run_name', '')

    # Create datalaoders
    train_loader, val_loader, test_loader = get_datalaoders(path_to_data,
                                                            config['batch_size'],
                                                            use_synthetic=config['use_synthetic'],
                                                            features=config['features'],
                                                            class_dict=config['class_dict'],
                                                            k_fold=config.get('k_fold', None))

    # Initialize model
    model = init_model(**config)

    trained_model, best_f1 = train(model, train_loader, val_loader, **config)

    model.load_state_dict(torch.load(trained_model))

    test(model, test_loader, **config)

    return best_f1


def run_experiment(path_to_data, num_experiments=6, **config):
    """ Automatically runs a forward chaining with all features.
        Runs num_experiments experiments sampled by Optimal Latin HyperCube per feature chain.

    Args:
        path_to_data (string): Path to the data directory
        num_experiments (int): Number of experiments to be created by latin hypercube
        **config: batch_size (int): Size of the batches
                  num_epochs (int): Max number of train epochs
                  seq_len (int): Length of model input time series after processing
                  criterion (torch.nn): Loss function
                  optim (torch.optim): Optimizer
                  optim_kwargs (dict): Optimizer kwargs
                  scheduler (torch.optim): LR scheduler for training
                  scheduler_kwargs (dict): Scheduler kwargs
                  early_stopping (int): Number of epchs or None for no early stopping
                  model_kwargs (dict): Model initialization parameters
                  class_dict (dict): Dictionary of desired classes
                  features (dict): Dictionary of val and train features as list
                  experiment_name (string): Name to store the experiment results or None
                  use_synthetic (bool): Use synthetic data for training
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
        'train': [RandomAugment(measurement_frequency=measurement_frequency)],
        'val': [RandomAugment(measurement_frequency=measurement_frequency, p=0)]
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

            run_f1_array[j] = best_run_f1

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


def run_k_fold(path_to_data, num_folds, **config):
    """ Does a k-fold experiment and stores the result of all 10 models

    Args:
        path_to_data (string): Path to the data directory
        num_folds (int): Number of folds
        **config: batch_size (int): Size of the batches
                  num_epochs (int): Max number of train epochs
                  seq_len (int): Length of model input time series after processing
                  criterion (torch.nn): Loss function
                  optim (torch.optim): Optimizer
                  optim_kwargs (dict): Optimizer kwargs
                  scheduler (torch.optim): LR scheduler for training
                  scheduler_kwargs (dict): Scheduler kwargs
                  early_stopping (int): Number of epchs or None for no early stopping
                  model_kwargs (dict): Model initialization parameters
                  class_dict (dict): Dictionary of desired classes
                  features (dict): Dictionary of val and train features as list
                  experiment_name (string): Name to store the experiment results or None
                  use_synthetic (bool): Use synthetic data for training
    """
    config['experiment_name'] = f'K_Fold_{datetime.now().time()}'.replace(':', '_')

    for fold_i in range(0, num_folds):
        config['run_name'] = f'fold_{fold_i}'
        config['k_fold'] = (fold_i, num_folds)
        run_config(path_to_data, **config)


def init_model(**config):
    """ Initializes model

    Args:
        **config: seq_len (int): Length of model input time series after processing
                  model_kwargs (dict): Model initialization parameters
                  class_dict (dict): Dictionary of desired classes
                  features (dict): Dictionary of val and train features as list
    Returns:
        (nn.Module): Model object
    """
    assert [config['model_kwargs']['name'] in ['CNN1D', 'LSTM', 'RESNET']]

    # Calculate input feature dimension for model initialization
    in_features = 0
    for feature in config['features']['train']:
        in_features += feature.feature_dim
    in_features = max(in_features, 1)

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

    return model


def get_datalaoders(path_to_data, batch_size, medal_id=None, use_synthetic=False, features=None, class_dict=TYPE_CLASS,
                    k_fold=None):
    """ Returns data loaders

    Args:
        path_to_data (string): Path to the dataset
        batch_size (int): Size of batches
        medal_id (int): 1-14 for single medal or None for all
        features (dict): Dict containing the train and val/test features
        use_synthetic (bool): Use synthetic data for training
        class_dict (dict): Dict of type class mapping
        k_fold (tuple): (fold_i (int), num_folds (int))

    Returns:
        train_loader (torch.utils.data.DataLoader)
        val_loader (torch.utils.data.DataLoader)
        test_loader (torch.utils.data.DataLoader)
    """
    num_workers = 2
    sampler = None

    train_set = BLOND(path_to_data=path_to_data,
                      fold='train',
                      transform=Compose(features['train']) if features is not None else features,
                      medal_id=medal_id,
                      use_synthetic=use_synthetic,
                      class_dict=class_dict,
                      k_fold=k_fold)
    val_set = BLOND(path_to_data=path_to_data,
                    fold='val',
                    transform=Compose(features['val']) if features is not None else features,
                    medal_id=medal_id,
                    use_synthetic=use_synthetic,
                    class_dict=class_dict,
                    k_fold=k_fold)
    test_set = BLOND(path_to_data=path_to_data,
                     fold='test',
                     transform=Compose(features['val']) if features is not None else features,
                     medal_id=medal_id,
                     use_synthetic=use_synthetic,
                     class_dict=class_dict,
                     k_fold=k_fold)

    if len(set(train_set.labels['Type'])) > 1:
        sampler = WeightedRandomSampler(train_set.labels['Weight'].values, len(train_set), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import os
    from src.utils import ROOT_DIR

    class_dict = {
        'Laptop': 0,
        'Monitor': 1,
        'USB Charger': 2
    }

    path = os.path.join(ROOT_DIR, 'data')

    t, _, _ = get_datalaoders(path, 10, medal_id=2)
