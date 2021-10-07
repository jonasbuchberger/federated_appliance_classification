import os
import datetime
import torch
import pandas as pd
import torch.distributed as dist
from src.models.experiment_utils import get_datalaoders, run_config
from src.models.test import test
from src.federated.federated_utils import receive_broadcast, send_gather, weighted_model_average
from src.utils import ROOT_DIR, SummaryWriter


class Client:

    def __init__(self, rank, world_size, backend='gloo', master_addr='127.0.0.1', master_port='29500',
                 path_to_data=None):
        """ Initializes the federated learning client

        Args:
            rank (int): Worker identifier, 0: server
            world_size (int): World size equals #clients + server
            backend (string): 'gloo' or 'nccl'
            master_addr (string): Ip address of server
            master_port (string): Port of the server
        """

        self.rank = rank
        self.world_size = world_size
        self.path_to_data = path_to_data
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.model_weights = torch.ones(world_size - 2)

        if self.path_to_data is None:
            self.path_to_data = os.path.join(ROOT_DIR, 'data')

    def init_process(self):
        # Setup for PyTorch connection
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port

        init_method = None
        if 'tcp' in self.master_addr:
            init_method = f'{self.master_addr}:{self.master_port}'

        dist.init_process_group(self.backend,
                                rank=self.rank,
                                world_size=self.world_size,
                                init_method=init_method,
                                timeout=datetime.timedelta(0, 10 * 1800),
                                )

    def run(self):
        # Receive train config from master
        config = receive_broadcast()

        # Get initial model
        model = receive_broadcast()

        path_to_data = os.path.join(ROOT_DIR, 'data')
        local_steps = config['epochs']['local_steps']
        aggregation_rounds = config['epochs']['agg_rounds']
        mode = config['epochs']['mode']
        criterion = config['criterion']
        optim = config['optim'](model.parameters(), **config['optim_kwargs'])
        scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])
        setting = config['setting']

        # Parameter assertions
        assert (setting == 'iid' or setting == 'noniid')
        assert (aggregation_rounds > 0 and local_steps > 0)

        train_loader, v, _ = get_datalaoders(path_to_data,
                                             config['batch_size'],
                                             use_synthetic=config['use_synthetic'],
                                             features=config['features'],
                                             medal_id=self.rank if setting == 'noniid' else None,
                                             r_split=(self.rank - 1,
                                                      self.world_size - 1) if setting == 'iid' else None,
                                             class_dict=config['class_dict'])

        # If steps are given train for #steps batches else train for full dataset
        if mode == 'epoch':
            steps = len(train_loader)
            local_epochs = local_steps
        elif mode == 'step':
            steps = local_steps
            local_epochs = 1
        else:
            raise ValueError

        # Prepare weight logging
        if config.get('weighted', False):
            weight_df = None
            log_path = os.path.join(ROOT_DIR,
                                    'models',
                                    config['experiment_name'],
                                    config.get('run_name', ''),
                                    f'test_{self.rank}')
            logger = SummaryWriter(log_path, filename_suffix='_train')
            os.makedirs(log_path, exist_ok=True)

        # Start local training
        for agg_i in range(0, aggregation_rounds):

            for epoch_l in range(0, local_epochs):

                epoch_train_loss = 0
                model = model.train()
                for _ in range(0, steps):
                    # Sample data
                    data = next(iter(train_loader))
                    x, y_target = data
                    optim.zero_grad()
                    y_pred = model(x)
                    loss = criterion(y_pred, y_target)
                    loss.backward()
                    optim.step()
                    epoch_train_loss += loss.item()
                epoch_train_loss /= len(train_loader)

                scheduler.step()

            # Calculate federated model with FedFomo weighting
            # https://arxiv.org/pdf/2012.08565.pdf
            if config.get('weighted', False):
                send_gather((self.rank, model))
                model_list = receive_broadcast()
                for model_i in model_list[:]:
                    if model_i[0] == self.rank:
                        model_list.remove(model_i)

                assert (len(model_list) == len(self.model_weights))
                epoch_val_loss = self._update_weights(model_list, model, criterion, v)
                model_list = [model_list[i][1] for i in torch.arange(len(model_list))[self.model_weights > 0]]
                model_list = [model] + model_list

                weight_df = pd.concat((weight_df, pd.Series(self.model_weights)), axis=1)
                weight_df.columns = torch.arange(len(weight_df.columns)).numpy()
                weight_df.T.to_csv(os.path.join(log_path, 'weight_log.csv'))

                if len(model_list) > 1:
                    model = weighted_model_average(model_list)

                model_dict = model.state_dict()
                for layer in model_dict.keys():
                    model_layer = model_dict[layer][:]
                    for i, (_, model_i) in enumerate(model_list):
                        if self.model_weights[i] > 0:
                            model_dict[layer] += self.model_weights[i] * (model_i.state_dict()[layer] - model_layer)
                model.load_state_dict(model_dict)

                # Log train and val losses
                logger.add_scalar('Loss/Train', epoch_train_loss, agg_i)
                logger.add_scalar('Loss/Validation', epoch_val_loss, agg_i)

            # Normal unweighted local training
            else:
                send_gather(model)
                model = receive_broadcast()

            # Update learning rate scheduler and optimizer with new model weights
            lr = optim.param_groups[0]['lr']
            scheduler_dict = scheduler.state_dict()

            optim = config['optim'](model.parameters(), lr, config['optim_kwargs']['weight_decay'])
            scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])
            scheduler.load_state_dict(scheduler_dict)

        # Transfer learning on the final global federated model
        if config.get('transfer', False):

            # Freeze all weights except classifier
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

            config['optim_kwargs']['lr'] = config['transfer_kwargs']['lr']
            config['optim_kwargs']['weight_decay'] = config['transfer_kwargs']['weight_decay']
            config['num_epochs'] = config['transfer_kwargs']['num_epochs']
            config['medal_id'] = self.rank

            exp_name = f'transfer_{self.rank}_ne-{config["num_epochs"]}_lr-{config["optim_kwargs"]["lr"]}'
            config['experiment_name'] = os.path.join(ROOT_DIR,
                                                     'models',
                                                     config['experiment_name'],
                                                     config.get('run_name', ''),
                                                     exp_name)

            config['run_name'] = ''
            run_config(os.path.join(ROOT_DIR, 'data'), **config)

            dist.barrier()

        # Test final federated model on private test set
        if config.get('local_test', False):
            _, _, test_loader = get_datalaoders(path_to_data,
                                                config['batch_size'],
                                                use_synthetic=config['use_synthetic'],
                                                features=config['features'],
                                                medal_id=self.rank if setting == 'noniid' else None,
                                                r_split=(self.rank - 1,
                                                         self.world_size - 1) if setting == 'iid' else None,
                                                class_dict=config['class_dict'])

            config['experiment_name'] = os.path.join(config['experiment_name'], config.get('run_name', ''))
            config['run_name'] = f'test_{self.rank}'

            test(model, test_loader, **config)

        if config.get('weighted', False):
            logger.close()

    def _update_weights(self, model_list, model, criterion, val_loader):
        """ Updates the weight vector of each model.

        Args:
            model_list (list): List of received models
            model (torch.nn.Module): Current client model
            criterion (torch.nn): Loss function
            val_loader (dataloader): Validation dataloader
        Returns:
            (float): The loss of the current model
        """
        num_val_batches = len(val_loader)
        model = model.eval()
        loss = 0

        # Calculate loss of current model
        with torch.no_grad():
            for data in val_loader:
                x, y_target = data
                loss += criterion(model(x), y_target)
            loss /= num_val_batches

            # Calculate loss of each model in model_list
            for i, (_, model_i) in enumerate(model_list):
                model_i = model_i.eval()
                with torch.no_grad():
                    loss_i = 0
                    for data in val_loader:
                        x, y_target = data
                        loss_i += criterion(model_i(x), y_target)
                    loss_i /= num_val_batches
                    self.model_weights[i] = (loss - loss_i) / self._model_norm(model, model_i)
                self.model_weights[self.model_weights < 0] = 0

        # Update weights
        if self.model_weights.sum() > 0:
            self.model_weights = self.model_weights / torch.sum(self.model_weights)
        self.model_weights[torch.isnan(self.model_weights)] = 0

        return loss

    def _model_norm(self, model_n, model_i):
        """ Calculates the norm of the difference of two models

        Args:
            model_n (nn.Module): Model one
            model_n (nn.Module): Model two
        Returns:
            (float): Norm of the model layers
        """
        model_dict = model_n.state_dict()
        layer_norms = []
        for layer in model_dict.keys():
            if 'num_batches_tracked' not in layer:
                model_dict[layer] -= model_i.state_dict()[layer]

                layer_norms.append(torch.linalg.norm(model_dict[layer]))

        norm = torch.mean(torch.as_tensor(layer_norms)) + 1e-5
        return norm
