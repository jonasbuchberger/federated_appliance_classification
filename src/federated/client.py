import os
from src.utils import ROOT_DIR
from src.models.experiment_utils import get_datalaoders
from src.federated.federated_utils import receive_broadcast, send_gather


def run_client(rank, size):
    # Receive train config from master
    config = receive_broadcast()
    # Get initial model
    model = receive_broadcast()

    device = 'cpu'
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
                                         #medal_id=rank,
                                         k_fold=(rank-1, size-1),
                                         class_dict=config['class_dict'])

    aggregation_rounds = int(total_epochs / local_epochs)
    for agg_i in range(0, aggregation_rounds):
        # print(f'Aggregation {agg_i}')

        for epoch_l in range(0, local_epochs):

            epoch_train_loss = 0
            model = model.train()
            for data in train_loader:
                x, y_target = data
                x, y_target = x.to(device), y_target.to(device)
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
