import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import ROOT_DIR
from src.visualization.confusion_matrix import tensor_confusion_matrix
from src.visualization.metrics_table import metrics_table


def test(model, test_loader, **config):
    """ Tests a given model and saves results to Tensorbaord

    Args:
        - model (torch.nn.Module): Pytorch model
        - test_loader (torch.utils.data.DataLoader): Data loader with test set
        - train_config (dict): Dictionary of train parameters
    """

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Process running on: {device}")

    class_dict = test_loader.dataset.class_dict
    experiment_name = config['experiment_name']
    run_name = config['run_name']
    log_path = os.path.join(ROOT_DIR, 'models', experiment_name, run_name)
    logger = SummaryWriter(log_path, filename_suffix='_test')
    os.makedirs(log_path, exist_ok=True)

    y_target_list = []
    y_pred_list = []

    model.to(device)
    model = model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y_target = data
            x, y_target = x.to(device), y_target.to(device)
            y_pred = model(x)
            _, y_pred = torch.max(y_pred, 1)
            y_pred_list.extend(y_pred.cpu().tolist())
            y_target_list.extend(y_target.cpu().tolist())

    results_df = metrics_table(y_target_list, y_pred_list, class_dict)
    results_df.to_csv(os.path.join(log_path, 'results.csv'))

    logger.add_scalar('Test/Averaged Precision', results_df['Precision'].iloc[-1])
    logger.add_scalar('Test/Averaged Recall', results_df['Recall'].iloc[-1])
    logger.add_scalar('Test/Averaged F1_Score', results_df['F1'].iloc[-1])
    logger.add_scalar('Test/Accuracy', results_df['Overall Accuracy'].iloc[-1])
    logger.add_image('Test/Confusion Matrix', tensor_confusion_matrix(y_target_list, y_pred_list, class_dict))
    logger.close()
