# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:13:22 2020

@author: cht15
"""
import torch
from tqdm import tqdm


def data_normalisation(data, norm_type='standard_deviation', association=False):
    """
    Normilisation through standard deviation.

    Late implemented rough normilisation of data with use of the standard
    deviation.

    Parameters
    ----------
    data : list
        List object containing the data (Different time series).
    norm_type : str, optional
        Str for optimisation method if futher implemented. # TODO
        The default is 'standard_deviation'.

    Returns
    -------
    mean : Tensor
        Sample mean along the '0' axis.
    standard_deviation : Tensor
        Sample standard deviation along the '0' axis.

    """
    N = 0
    # Mean
    _inter_sum_data = 0
    for i in tqdm(range(len(data)), desc='mean'):
        N += data[i].shape[0]  # Number of datapoints accross all time series
        _inter_sum_data += data[i].sum(axis=0)  # Summation of all time series
    mean = _inter_sum_data/N  # Sum of data devided by number of datapoints

    # STD
    _inter_std_step = 0
    for j in tqdm(range(len(data)), desc='STD'):
        _inter_std_step += ((data[i]-mean)**2).sum(axis=0)  # Step for STD
    standard_deviation = torch.sqrt(_inter_std_step/(N-1))  # Unbiased (N-1)
# =============================================================================
# The following code is only usefull for the specific data structure used
# in the given project, as some of these variables are categorical, they aren't
# standardised. Faster to do them all and then ignore than only do parts.
# =============================================================================
    doi = ['length', 'data_length', 'phy', 'rate', 'mcs1', 'totalload', 'up',
           'inter_arrival']
    categorical = ['mcs1', 'phy', 'up']
    if association is True:
            doi = ['time', 'length', 'data_length', 'phy', 'rate', 'msc1',
                   'totalload', 'up', 'association', 'inter_arrival']
            categorical = ['mcs1', 'phy', 'up', 'time', 'association']
    for i in range(len(doi)):
        if doi[i] in categorical:  #Check if the given column is categorical.
            mean[i] = 0
            standard_deviation[i] = 1

    # Normalise
    for i in tqdm(range(len(data)), desc='norm'):
        data[i] = (data[i]-mean)/standard_deviation
    return data
