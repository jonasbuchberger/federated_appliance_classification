import os
import json
import time
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from src.utils import ROOT_DIR, SummaryWriter
from src.models.test import test
from src.models.experiment_utils import init_model, get_datalaoders
from src.federated.federated_utils import receive_gather, send_broadcast, weighted_model_average


def run_server(size, config):
    feature_string = '['
    for feature in config['features']['train']:
        feature_string += f'{feature.__class__.__name__}_'
    feature_string = feature_string[:-1] + ']'

    if config['experiment_name'] is None:
        config['experiment_name'] = f"Federated_{size}_" \
                                    f"{config['model_kwargs']['name']}_" \
                                    f"{config['optim'].__name__}_" \
                                    f"{config['criterion'].__class__.__name__}_" \
                                    f"CLASS_{len(config['class_dict'].keys())}_" \
                                    f"{feature_string}_6400_Synthetic"

        config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                             f"wd-{config['optim_kwargs']['weight_decay']}_" \
                             f"nl-{config['model_kwargs']['num_layers']}_" \
                             f"ss-{config['model_kwargs']['start_size']}"

    path_to_data = os.path.join(ROOT_DIR, 'data')
    _, val_loader, test_loader = get_datalaoders(path_to_data,
                                                 config['batch_size'],
                                                 use_synthetic=config['use_synthetic'],
                                                 features=config['features'],
                                                 class_dict=config['class_dict'])

    total_epochs = config['total_epochs']
    local_epochs = config['local_epochs']
    criterion = config['criterion']
    experiment_name = config['experiment_name']
    run_name = config['run_name']

    log_path = os.path.join(ROOT_DIR, 'models', experiment_name, run_name)
    logger = SummaryWriter(log_path, filename_suffix='_train')
    os.makedirs(log_path, exist_ok=True)

    # Broadcast train config to all clients
    print('Send train config to clients.')
    send_broadcast(config)

    model = init_model(**config)
    # Broadcast initial model
    print('Send model to clients.')
    send_broadcast(model)

    start = time.time()
    aggregation_rounds = int(total_epochs / local_epochs)
    for i_agg in tqdm(range(0, aggregation_rounds)):

        # Gather trained models
        model_list = receive_gather(size)

        # AVG models
        model = weighted_model_average(model_list)

        # Validate aggregated model
        val(val_loader, model, criterion, i_agg, logger)

        # Broadcast new model
        send_broadcast(model)

    print(f'Finished Training: {(time.time() - start) / 60}min')

    test(model, test_loader, **config)
    print('Finished Testing')


def val(val_loader, model, criterion, i_agg, logger, device='cpu'):

    num_correct = 0
    num_samples = 0
    y_target_list = []
    y_pred_list = []
    epoch_val_loss = 0
    num_val_batches = len(val_loader)
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            x, y_target = data
            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_target)
            epoch_val_loss += loss.item()
            _, y_pred = torch.max(y_pred, 1)
            y_pred_list.extend(y_pred.cpu().tolist())
            y_target_list.extend(y_target.cpu().tolist())
            num_correct += torch.sum(y_pred == y_target).item()
            num_samples += y_target.size(0)
    epoch_val_loss /= num_val_batches
    epoch_val_accuracy = num_correct / num_samples

    # Compute precision, recall F1-score and support for validations set
    epoch_precision, epoch_recall, epoch_f1, _ = precision_recall_fscore_support(y_target_list, y_pred_list,
                                                                                 average="macro")

    logger.add_scalar('Validation/Precision', epoch_precision, i_agg)
    logger.add_scalar('Validation/Recall', epoch_recall, i_agg)
    logger.add_scalar('Validation/F1_Score', epoch_f1, i_agg)
    logger.add_scalar('Validation/Accuracy', epoch_val_accuracy, i_agg)
    logger.add_scalar('Loss/Validation', epoch_val_loss, i_agg)