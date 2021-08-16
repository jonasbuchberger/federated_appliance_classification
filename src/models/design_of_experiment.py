#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: René Schwermer
E-Mail: rene.schwermer@tum.de
Date: 05.11.2020
Project: FOAM
Version: 0.0.1
"""
import numpy as np
from smt.sampling_methods import LHS


def lh(boundaries: np.array,
       nr_of_samples: int) -> np.array:
    """
    Design of experiment with latin hypercube sampling. Gives an evenly
    distributed sampling withing the given boundaries for each variable.
    The amount of given sampling points are evenly distributed. This
    methods is similar to a Monte Carlo sampling.
    The return samples is a m x n matrix with m = nr_of_samples and
    n = nr of variables/boundaries.

    Example:
    1000x3 gives (1000, 1) with the values for the first variable
    as a np.array, (1000, 2) for the 2nd, and (1000, 3) for the 3rd
    variable.

    :param boundaries: np.array, | boundaries for each parameter to
                                 | consider,
                                 | e.g. [[0.0, 2.0], [1.5, 2.0]]
    :param nr_of_samples: int, number of total samples
    :return: np.array, samples withing the variable space
    """
    boundaries = np.array(boundaries)
    sampling = LHS(xlimits=boundaries, criterion='ese')
    samples = sampling(nr_of_samples)

    return samples


def to_dict(samples: np.array) -> dict:
    """
    Creates a dictionary with "exp_i" as key and one value per variable
    as value. The input "samples" consists of all values for one
    variable/boundary as a np.array. For later using the design of
    experiment, for each "exp_i" one value from each variable is taken.
    The order is not changed.

    Example:
    Parameters for the experiment:
    velocity 1 [m/s]    0 <= v_1 <= 2
    temperature 1 [°C]  100 <= T_1 <= 400
    pressure 1 [bar]    0.1 <= p_1 <= 0.4

                   v_1          T_1               p_1
    samples = [[0, 1, 2], [100, 400, 300], [0.1, 0.4, 0.3]] gives
    doe = {"exp_1": [0, 100, 0.1],
           "exp_2": [1, 400, 0.4],
           "exp_3": [2, 300, 0.3]}

    :param samples: np.array, samples withing a variable space
    :return: dict, ordered design of experiment
    """
    values = [None] * samples.shape[1]

    for i in range(0, samples.shape[1], 1):
        values[i] = samples[:, i]

    doe = {}
    for i in range(0, samples.shape[0], 1):
        entries = []
        for j in range(0, samples.shape[1], 1):
            entries.append(values[j][i])

        doe["exp_" + str(i)] = entries

    return doe
