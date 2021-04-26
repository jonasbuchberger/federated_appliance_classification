import os
import warnings

from src.data.dataset_blond import get_datalaoders
from src.models.models import BlondConvNet
from src.models.train import train
from src.models.test import test
from src.utils import ROOT_DIR

warnings.filterwarnings("ignore", category=UserWarning)
from src.features.features import *

if __name__ == '__main__':
    path_to_data = os.path.join(ROOT_DIR, 'data')
    class_dict = {
        'Laptop': 0,
        'Monitor': 1,
        'USB Charger': 2
    }
    features = {
        'train': ACPower(),
        'val': ACPower()
    }
    config = {
        'batch_size': 60,
        'num_epochs': 30,
        'criterion': torch.nn.CrossEntropyLoss(),
        'optim': torch.optim.SGD,
        'optim_kwargs': {'lr': 0.001, 'weight_decay': 0.0},
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'scheduler_kwargs': {'factor': 0.1, 'patience': 3, 'mode': 'max'},
        'early_stopping': 5,
        'experiment_name': 'test_ac',
    }

    train_loader, val_loader, test_loader = get_datalaoders(path_to_data, config['batch_size'], features=features,
                                                            class_dict=class_dict)

    model = BlondConvNet(in_features=4,
                         seq_len=25,
                         num_classes=len(class_dict.keys()),
                         num_layers=2)

    trained_model = train(model, train_loader, val_loader, **config)

    model.load_state_dict(torch.load(trained_model))

    test(model, test_loader, **config)
