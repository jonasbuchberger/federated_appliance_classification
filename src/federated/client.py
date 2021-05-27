import os
import torch.distributed as dist
from src.utils import ROOT_DIR
from src.models.experiment_utils import get_datalaoders
from src.federated.federated_utils import receive_broadcast, send_gather


class Client:

    def __init__(self, rank, world_size, backend='gloo', master_addr='127.0.0.1', master_port='29500', path_to_data=None):
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

        dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size, init_method=init_method)

    def run(self):
        # Receive train config from master
        config = receive_broadcast()
        # Get initial model
        model = receive_broadcast()

        path_to_data = os.path.join(ROOT_DIR, 'data')
        local_epochs = config['local_epochs']
        total_epochs = config['total_epochs']
        criterion = config['criterion']
        optim = config['optim'](model.parameters(), **config['optim_kwargs'])
        scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])

        train_loader, _, _ = get_datalaoders(path_to_data,
                                             config['batch_size'],
                                             use_synthetic=config['use_synthetic'],
                                             features=config['features'],
                                             medal_id=self.rank,
                                             #k_fold=(self.rank-1, self.world_size-1),
                                             class_dict=config['class_dict'])

        aggregation_rounds = int(total_epochs / local_epochs)
        for agg_i in range(0, aggregation_rounds):
            # print(f'Aggregation {agg_i}')

            for epoch_l in range(0, local_epochs):

                epoch_train_loss = 0
                model = model.train()
                for data in train_loader:
                    x, y_target = data
                    optim.zero_grad()
                    y_pred = model(x)
                    loss = criterion(y_pred, y_target)
                    loss.backward()
                    optim.step()
                    epoch_train_loss += loss.item()
                epoch_train_loss /= len(train_loader)

                scheduler.step(epoch_train_loss)

            # Send trained model
            send_gather(model)

            model = receive_broadcast()
            lr = optim.param_groups[0]['lr']
            optim = config['optim'](model.parameters(), lr, config['optim_kwargs']['weight_decay'])
            scheduler = config['scheduler'](optim, **config['scheduler_kwargs'])
