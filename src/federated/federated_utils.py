from torch import nn
import torch.distributed as dist


def send_broadcast(obj):
    """ Send an object from the server to all clients.
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
        weight_list (list): List of weights for each model. Weights should sum up to 1.0

    Returns:
        (nn.Module): The new aggregated model
    """
    if weight_list is not None:
        assert (weight_list.sum() == 1.0)

    model_dict = model_list[0].state_dict()
    """
    TODO: Weighted averaging
    """
    for layer in model_dict.keys():
        for model in model_list[1:]:
            model_dict[layer] += model.state_dict()[layer]
        model_dict[layer] = model_dict[layer] / float(len(model_list))

    model_list[0].load_state_dict(model_dict)
    return model_list[0]
