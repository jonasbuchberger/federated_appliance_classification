import os
import datetime
import torch.distributed as dist
from src.utils import ROOT_DIR
from src.models.experiment_utils import get_datalaoders, run_config
from src.federated.federated_utils import receive_broadcast, send_gather


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

        if self.path_to_data is None:
            self.path_to_data = os.path.join(ROOT_DIR, 'data')

    def init_process(self):

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
        # scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])
        setting = config['setting']

        # Parameter assertions
        assert (setting == 'iid' or setting == 'noniid')
        assert (aggregation_rounds > 0 and local_steps > 0)

        train_loader, _, _ = get_datalaoders(path_to_data,
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

                # scheduler.step(epoch_train_loss)

            # Send trained model
            send_gather(model)

            model = receive_broadcast()
            lr = optim.param_groups[0]['lr']
            # scheduler_dict = scheduler.state_dict()

            optim = config['optim'](model.parameters(), lr, config['optim_kwargs']['weight_decay'])
            # scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])
            # scheduler.load_state_dict(scheduler_dict)

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
