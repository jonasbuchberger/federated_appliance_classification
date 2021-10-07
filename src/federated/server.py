import os
import json
import torch
from datetime import timedelta
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

        self.model_weights = torch.ones(self.world_size - 1)

        if self.path_to_data is None:
            self.path_to_data = os.path.join(ROOT_DIR, 'data')

    def init_process(self):

        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port

        init_method = None
        if 'tcp' in self.master_addr:
            init_method = f'{self.master_addr}:{self.master_port}'

        timeout = max([self.config['epochs']['local_steps'],
                       self.config.get('transfer_kwargs', {}).get('num_epochs', 0)])
        timeout = 5

        dist.init_process_group(self.backend,
                                rank=0,
                                world_size=self.world_size,
                                init_method=init_method,
                                timeout=timedelta(0, timeout * 1800),
                                )

    def _setup_experiment(self, config):
        """ Setup for experimental setting

        Args:
            config (dict): Training configuration
        """

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
                                        f"{feature_string}"

            if config['use_synthetic']:
                config['experiment_name'] = f"{config['experiment_name']}_Syn"

            config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                                 f"wd-{config['optim_kwargs']['weight_decay']}_" \
                                 f"nl-{config['model_kwargs']['num_layers']}_" \
                                 f"ss-{config['model_kwargs']['start_size']}_" \
                                 f"agg-{config['epochs']['agg_rounds']}_" \
                                 f"{config['epochs']['mode']}-{config['epochs']['local_steps']}_" \
                                 f"{config['setting']}"
        return config

    def run(self):
        aggregation_rounds = self.config['epochs']['agg_rounds']
        criterion = self.config['criterion']
        experiment_name = self.config['experiment_name']
        run_name = self.config.get('run_name', '')
        early_stopping = self.config.get('early_stopping', None)
        patience = early_stopping
        logging_factor = self.config.get('logging_factor', 1)

        # Parameter assertions
        assert (aggregation_rounds > 0)
        assert (early_stopping is None or early_stopping > 0)

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

        best_f1 = 0
        early_stopping_val_loss = None
        for i_agg in tqdm(range(0, aggregation_rounds)):
            # Gather trained models
            model_list = receive_gather(self.world_size)
            if self.config.get('weighted', False):
                send_broadcast(model_list)
            else:
                # AVG models
                model = weighted_model_average(model_list, self.model_weights)

                # Validate aggregated model
                if (i_agg % logging_factor == 0 or i_agg == aggregation_rounds - 1) and logging_factor != -1:
                    epoch_val_loss, epoch_f1 = self._val(val_loader, model, criterion, i_agg, logger)

                    if best_f1 is None or best_f1 < epoch_f1:
                        torch.save(model.state_dict(), f"{log_path}/model.pth")
                        best_f1 = epoch_f1

                    # Early stopping
                    if early_stopping is not None:
                        if early_stopping_val_loss is not None and early_stopping_val_loss < epoch_val_loss:
                            patience -= 1
                        else:
                            early_stopping_val_loss = epoch_val_loss
                            patience = early_stopping
                        if patience <= 0:
                            dist.destroy_process_group()
                            break

                # Broadcast new model
                send_broadcast(model)

        print('Finished Aggregating')

        # Try to load the best saved checkpoint else test current model
        try:
            model.load_state_dict(torch.load(f"{log_path}/model.pth"))
        except FileNotFoundError:
            torch.save(model.state_dict(), f"{log_path}/model.pth")

        test(model, test_loader, **self.config)

        # Save train parameters and model architecture
        with open(os.path.join(log_path, 'params.json'), 'w') as text_file:
            json.dump(self.config, text_file, indent=4, default=str)
        text_file.close()
        with open(os.path.join(log_path, 'model.txt'), 'w') as text_file:
            text_file.write(str(model))
        text_file.close()

        print('Finished Testing')

        # If transfer is True wait for all pis to finish to measure usage
        if self.config.get('transfer', False):
            print('Waiting for all clients finished transferring')
            torch.distributed.barrier()

        return log_path

    def _val(self, val_loader, model, criterion, i_agg, logger):
        """ Calculates the validation metrics of a model checkpoint

        Args:
            val_loader (dataloader): Validation dataloader
            model (torch.nn.Module): Current global model
            criterion (torch.nn): Loss function
            i_agg (int): Current aggregation round
            logger (torch.utils.tensorboard): Tensorboard logger

        Returns:
            (nn.Module): The new aggregated model
        """
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

        return epoch_val_loss, epoch_f1


