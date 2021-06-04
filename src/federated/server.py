import os
import json
import time
import torch
import datetime
from tqdm import tqdm
import torch.distributed as dist
from sklearn.metrics import precision_recall_fscore_support
from src.utils import ROOT_DIR, SummaryWriter
from src.models.test import test
from src.models.experiment_utils import init_model, get_datalaoders
from src.federated.federated_utils import receive_gather, send_broadcast, weighted_model_average


class Server:

    def __init__(self, world_size, config, path_to_data=None, backend='gloo'):
        """ Initializes the federated learning server

        Args:
            world_size (int): World size equals #clients + server
            config: Train config
            path_to_data: Path to data directory, if None standard path is used
            backend (string): 'gloo' or 'nccl'
        """

        self.world_size = world_size
        self.backend = backend
        self.config = self._setup_experiment(config)
        self.path_to_data = path_to_data
        self.master_addr = '127.0.0.1'
        self.master_port = '29500'

        if self.path_to_data is None:
            self.path_to_data = os.path.join(ROOT_DIR, 'data')

    def init_process(self):

        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port

        init_method = None
        if 'tcp' in self.master_addr:
            init_method = f'{self.master_addr}:{self.master_port}'

        dist.init_process_group(self.backend,
                                rank=0,
                                world_size=self.world_size,
                                init_method=init_method,
                                #timeout=datetime.timedelta(0, 60),
                                )

    def _setup_experiment(self, config):

        feature_string = '['
        for feature in config['features']['train']:
            feature_string += f'{feature.__class__.__name__}_'
        feature_string = feature_string[:-1] + ']'

        if config['experiment_name'] is None:
            config['experiment_name'] = f"Federated_{self.world_size}_" \
                                        f"{config['model_kwargs']['name']}_" \
                                        f"{config['optim'].__name__}_" \
                                        f"{config['criterion'].__class__.__name__}_" \
                                        f"CLASS_{len(config['class_dict'].keys())}_" \
                                        f"{feature_string}_6400_Synthetic"

            config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                                 f"wd-{config['optim_kwargs']['weight_decay']}_" \
                                 f"nl-{config['model_kwargs']['num_layers']}_" \
                                 f"ss-{config['model_kwargs']['start_size']}"

        return config

    def run(self):

        total_epochs = self.config['total_epochs']
        local_epochs = self.config['local_epochs']
        criterion = self.config['criterion']
        experiment_name = self.config['experiment_name']
        run_name = self.config['run_name']

        _, val_loader, test_loader = get_datalaoders(self.path_to_data,
                                                     self.config['batch_size'],
                                                     use_synthetic=self.config['use_synthetic'],
                                                     features=self.config['features'],
                                                     class_dict=self.config['class_dict'])

        log_path = os.path.join(ROOT_DIR, 'models', experiment_name, run_name)
        logger = SummaryWriter(log_path, filename_suffix='_train')
        os.makedirs(log_path, exist_ok=True)

        # Broadcast train config to all clients
        print('Send train config to clients.')
        send_broadcast(self.config)

        model = init_model(**self.config)
        # Broadcast initial model
        print('Send model to clients.')
        send_broadcast(model)

        start = time.time()
        aggregation_rounds = int(total_epochs / local_epochs)
        for i_agg in tqdm(range(0, aggregation_rounds)):
            # Gather trained models
            model_list = receive_gather(self.world_size)

            # AVG models
            model = weighted_model_average(model_list)

            # Validate aggregated model
            self._val(val_loader, model, criterion, i_agg, logger)

            # Broadcast new model
            send_broadcast(model)

        print(f'Finished Training: {(time.time() - start) / 60}min')

        test(model, test_loader, **self.config)

        # Save train parameters and model architecture
        with open(os.path.join(log_path, 'params.json'), 'w') as text_file:
            json.dump(self.config, text_file, indent=4, default=str)
        text_file.close()
        with open(os.path.join(log_path, 'model.txt'), 'w') as text_file:
            text_file.write(str(model))
        text_file.close()

        print('Finished Testing')

    def _val(self, val_loader, model, criterion, i_agg, logger):

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
