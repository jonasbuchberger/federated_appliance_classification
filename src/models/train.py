import json
import os

import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from src.utils import ROOT_DIR
from src.utils import SummaryWriter


def train(model, train_loader, val_loader, **config):
    """
    Args:
        model (torch.nn.Module): Pytorch model
        train_loader (torch.utils.data.DataLoader): Dataloader with training set
        val_loader (torch.utils.data.DataLoader): Dataloader with validation set
        config (dict): Dictionary of train parameters
    Returns:
        (string): Path of the trained model
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    # Getting parameters from train_config
    early_stopping = config.get('early_stopping', None)
    experiment_name = config['experiment_name']
    run_name = config['run_name']
    num_epochs = config['num_epochs']
    criterion = config['criterion']
    optim = config['optim'](model.parameters(), **config['optim_kwargs'])
    scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])

    log_path = os.path.join(ROOT_DIR, 'models', experiment_name, run_name)
    logger = SummaryWriter(log_path, filename_suffix='_train')
    os.makedirs(log_path, exist_ok=True)

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    best_acc = None
    best_f1 = None
    early_stopping_val_loss = None

    model.to(device)
    criterion.to(device)
    for i_epoch in range(num_epochs):

        epoch_train_loss = 0
        epoch_val_loss = 0

        model = model.train()
        for data in tqdm(train_loader):
            x, y_target = data
            x, y_target = x.to(device), y_target.to(device)
            optim.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optim.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= num_train_batches

        num_correct = 0
        num_samples = 0
        y_target_list = []
        y_pred_list = []

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

        logger.add_scalar('Validation/Precision', epoch_precision, i_epoch)
        logger.add_scalar('Validation/Recall', epoch_recall, i_epoch)
        logger.add_scalar('Validation/F1_Score', epoch_f1, i_epoch)
        logger.add_scalar('Validation/Accuracy', epoch_val_accuracy, i_epoch)
        logger.add_scalar('Loss/Train', epoch_train_loss, i_epoch)
        logger.add_scalar('Loss/Validation', epoch_val_loss, i_epoch)

        if best_f1 is None or best_f1 < epoch_f1:
            torch.save(model.state_dict(), f"{log_path}/model.pth")
            best_f1 = epoch_f1
            best_acc = epoch_val_accuracy

        if early_stopping is not None and i_epoch % early_stopping == 0:
            if early_stopping_val_loss is not None and early_stopping_val_loss < epoch_val_loss:
                break
            else:
                # torch.save(model.state_dict(), f"{log_path}/checkpoint_{i_epoch}.pth")
                early_stopping_val_loss = epoch_val_loss

        scheduler.step(epoch_f1)

    logger.add_graph(model.cpu(), x[0].unsqueeze(0))
    logger.add_hparams({'lr': config['optim_kwargs']['lr'],
                        'weight_decay': config['optim_kwargs']['weight_decay'],
                        'batch_size': config['batch_size'],
                        'num_layers': config['model_kwargs']['num_layers'],
                        'start_size': config['model_kwargs']['start_size']},
                       {'Hparam/Accuracy': best_acc,
                        'Hparam/F1': best_f1})
    logger.close()

    # Save train parameters and model architecture
    with open(os.path.join(log_path, 'params.json'), 'w') as text_file:
        json.dump(config, text_file, indent=4, default=str)
    text_file.close()
    with open(os.path.join(log_path, 'model.txt'), 'w') as text_file:
        text_file.write(str(model))
    text_file.close()

    return os.path.join(log_path, 'model.pth')
