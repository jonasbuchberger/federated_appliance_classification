import os
import time
from src.utils import ROOT_DIR
from src.models.test import test
from src.models.experiment_utils import init_model, get_datalaoders
from src.federated.federated_utils import receive_gather, send_broadcast, weighted_model_average


def run_server(size, config):

    feature_string = '['
    for feature in config['features']['train']:
        feature_string += f'{feature.__class__.__name__}_'
    feature_string = feature_string[:-1] + ']'

    if config['experiment_name'] is None:
        config['experiment_name'] = f"Federated_" \
                                    f"{config['model_kwargs']['name']}_" \
                                    f"{config['optim'].__name__}_" \
                                    f"{config['criterion'].__class__.__name__}_" \
                                    f"CLASS_{len(config['class_dict'].keys())}_" \
                                    f"{feature_string}_6400_Synthetic"

        config['run_name'] = f"lr-{config['optim_kwargs']['lr']}_" \
                             f"wd-{config['optim_kwargs']['weight_decay']}_" \
                             f"nl-{config['model_kwargs']['num_layers']}_" \
                             f"ss-{config['model_kwargs']['start_size']}"

    total_epochs = config['total_epochs']
    local_epochs = config['local_epochs']

    # Broadcast train config to all clients
    print('Send train config to clients.')
    send_broadcast(config)

    model = init_model(**config)
    # Broadcast initial model
    print('Send model to clients.')
    send_broadcast(model)

    start = time.time()
    aggregation_rounds = int(total_epochs / local_epochs)
    for agg_i in range(0, aggregation_rounds):
        print(f'Aggregation {agg_i}')

        # Gather trained models
        model_list = receive_gather(size)

        # AVG models
        model = weighted_model_average(model_list)

        # Broadcast new model
        send_broadcast(model)

    print(f'Finished Training: {start - time.time()}s')
    path_to_data = os.path.join(ROOT_DIR, 'data')
    _, _, test_loader = get_datalaoders(path_to_data,
                                        config['batch_size'],
                                        use_synthetic=config['use_synthetic'],
                                        features=config['features'],
                                        class_dict=config['class_dict'])
    test(model, test_loader, **config)
    print('Finished Testing')
