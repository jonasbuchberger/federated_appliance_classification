import os
import torch
from torch import nn
import torch.distributed as dist
import pandas as pd
from torch.nn.functional import softmax
from datetime import datetime, timedelta


def send_broadcast(obj):
    """ Send an object from the server to all clients
        Synchronous: Waits for all clients to have sent object received

    Args:
        obj (Any): Object to send
    """
    dist.broadcast_object_list([obj], src=0)


def receive_broadcast():
    """ Client receives the broadcast from the server

    Returns:
        (Any): The sent object
    """
    obj_list = [nn.Identity()]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def send_gather(obj):
    """ Client sends object to the server

    Args:
        obj (Any): Object to send to the server
    """
    dist.gather_object(obj, None, dst=0)


def receive_gather(size):
    """ Server receives one object from each client

    Args:
        size: Number of clients in setup (incl. server)

    Returns:
        (list): List of length size-1 containing the objects sent by the clients
    """
    obj_list = [None] * size
    dist.gather_object(None, obj_list, dst=0)
    return obj_list[1:]


def weighted_model_average(model_list, weight_list=None):
    """ Calculates the average or weighted average of all given models

    Args:
        model_list (list): List of models
        weight_list (list): List of weights for each model.

    Returns:
        (nn.Module): The new aggregated model
    """
    model_dict = model_list[0].state_dict()
    if weight_list is not None:
        weight_list = softmax(torch.as_tensor(weight_list, dtype=torch.float), dim=0)

        for layer in model_dict.keys():
            model_dict[layer] = model_dict[layer] * weight_list[0]

        for layer in model_dict.keys():
            for i, model in enumerate(model_list[1:]):
                model_dict[layer] += model.state_dict()[layer] * weight_list[i+1]

    else:
        for layer in model_dict.keys():
            for model in model_list[1:]:
                model_dict[layer] += model.state_dict()[layer]
            model_dict[layer] = model_dict[layer] / float(len(model_list))

    model_list[0].load_state_dict(model_dict)
    return model_list[0]


def get_pi_usage(start_time, end_time, dest_path):
    """ Gets pi usage information from postgressSQL database.

    Args:
        start_time (datetime): Start of training
        end_time (datetime): End of testing
        dest_path (string): Path to destination of logs
    """
    import psycopg2

    connection = psycopg2.connect(host="131.159.52.93",
                                  port=5432,
                                  user="jonas",
                                  password="jonas04",
                                  database="rpc")

    os.makedirs(dest_path, exist_ok=True)

    start_time = start_time - timedelta(seconds=5)
    for pi in [17, 18, 19, 20, 21, 22, 23, 24, 41, 42, 43, 45, 46, 47, 48]:
        sql = f"SELECT time, memory_used, bytes_sent, bytes_recv, packets_recv, packets_sent, " \
              f"cpu0_user, cpu1_user, cpu2_user, cpu3_user, " \
              f"cpu0_freq , cpu1_freq, cpu2_freq, cpu3_freq " \
              f"FROM raspi{pi};"

        data = pd.read_sql_query(sql, connection).set_index('time')
        data = data[str(start_time): str(end_time)]
        data.to_csv(f'{dest_path}/raspi{pi}.csv')

    connection.close()


if __name__ == '__main__':
    model = nn.Linear(300, 300)
    model_list = [model, model, model, model]
    weight_list = [1, 2, 3, 4]
    model = weighted_model_average(model_list, weight_list)
