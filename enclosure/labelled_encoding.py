#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:33:34 2020

@author: cht
"""

import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from itertools import permutations

from tqdm import tqdm
from pathlib import Path
from label_dict import label_dict
from bin_label_dict import bin_label_dict
from pytorch_modules import GRUEncoder, FullyConnected
from rough_normilisation import data_normalisation


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# %% Setup
if __name__ == "__main__":
    path = Path('/data/')

    n_labels = 5  # number of different labels
    layers = 2  # number of layers trained in the autoencoder
    if n_labels == 2:
        label_dict = bin_label_dict

    data = []
    models_dict = {}
    all_packet_labels = []
    u_association = []
    os.chdir(path)
    # The following magic numbers are from the creation of data in word group.
    time_index = 0  # First index is time index
    association_index = -2 # second last index is association
    doi = np.ones(10, dtype=bool)
    doi[time_index] = False
    doi[association_index] = False
    # Magic Encoded mac, see mac_encoded.npy (Mac refference)
    mac_encoded = [81343, 82447, 66391, 67314]

    # %% Load data
    # All data in directory
    for f in os.listdir():
        with open(f, 'rb') as fp:
            all_words = pickle.load(fp)  # Load pickled data (torch)
        data += all_words  # group all data together (if multiple files)
        _labels = []
        if f.split(sep='_')[0] == 'labeled':
            label_key = int(f.split(sep='_')[-1].split(sep='.')[0])
            labels = label_dict[label_key]
            for i in tqdm(range(len(all_words)), desc=f):
                association = all_words[i][0][-2]
                n_words = len(all_words[i])
                packet_labels = np.zeros((n_words, n_labels))
                for j in range(n_words):
                    time = all_words[i][j][0]
                    for label in labels:
                        if label[-1] == 0 and association in mac_encoded:
                            if label[0] > 0:
                                packet_labels[j][int(label[-2])] += \
                                    time < label[1] and time >= label[0]
                            else:  # Catch negative time labeled packets.
                                packet_labels[j][int(label[-2])] += \
                                    time < label[1]
                        elif label[-1] == 1 and association in mac_encoded[2:]:
                            if label[0] > 0:
                                packet_labels[j][int(label[-2])] += \
                                    time < label[1] and time >= label[0]
                            else:  # Catch negative time labeled packets.
                                packet_labels[j][int(label[-2])] += \
                                    time < label[1]
                        elif label[-1] == 2 and association in mac_encoded[:2]:
                            if label[0] > 0:
                                packet_labels[j][int(label[-2])] += \
                                    time < label[1] and time >= label[0]
                            else:  # Catch negative time labeled packets.
                                packet_labels[j][int(label[-2])] += \
                                    time < label[1]
                _labels.append(packet_labels)
            all_packet_labels += _labels
        else:
            for i in range(len(all_words)):
                _labels.append(np.zeros(1))
            all_packet_labels += _labels
            

    data = data_normalisation(data, association=True)
    to_pd = np.concatenate(data)
    
    doi = ['time', 'length', 'data_length', 'phy', 'rate', 'msc1', 'totalload',
           'up', 'association', 'inter_arrival']
    doi_bool = np.ones_like(doi, dtype=bool)
    doi_bool[0] = False
    doi_bool[-2] = False

    possible_models = os.listdir('models')  # Find all trained models
    # Find the unique specification for trained models through their names.
    models_specs = np.unique(np.array([models.split(sep='_')[-2] + '_' 
                              + models.split(sep='_')[-1].split(sep='.')[0]
                                  for models in possible_models]))

    # %% Pick a trained model
    for i in range(len(models_specs)):
        models_dict[i] = models_specs[i]
    print(f'The trained models contain the following "Neuron_epoch" combinations.')
    print(f'{models_dict}')
    print(f'Chose, by the corresponding number.')
    try:
        key = int(input('Model nr: '))
        if key not in models_dict:
            print("Please chose a valid interger model number")
            key = int(input('Model nr: '))
    except ValueError:
        print("Please chose a valid interger model number")
        key = int(input('Model nr: '))
        if key not in models_dict:
            print("Please chose a valid interger model number")
            key = int(input('Model nr: '))
    if key not in models_dict:
        raise ValueError(f'The chosen model nr. "{key}" isnt a valid option.')
    # key = 0

    hidden_neurons = int(models_dict[key].split(sep='_')[0])
    encoder_model = GRUEncoder(hidden_neurons, layers)
    center_model = FullyConnected(hidden_neurons)

    # %% Encoding
    encoder_model.load_state_dict(
        torch.load('models/new_encoder_'+models_dict[key]+'.pt'))
    center_model.load_state_dict(
        torch.load('models/new_center_'+models_dict[key]+'.pt'))
    encoder_model.eval()
    center_model.eval()
    n_labeled_series = len(all_packet_labels)
    # encoded = np.empty((n_labeled_series , hidden_neurons*layers))
    data_encoded = np.empty((len(data) , hidden_neurons*layers))

    labelled_data = []
    true_labels = []
    # increment = 0
    i=0
    for j in tqdm(range(len(data)), desc=f'Encoding:'):
        series = torch.reshape(data[j].T[doi_bool].T, (1, -1, 8))
        hidden = encoder_model(series)
        center = center_model(hidden)
        center = torch.transpose(center, 0, 1)
        if len(all_packet_labels[j].shape) == 1:
            data_encoded[j][:] = torch.flatten(center[-1]).detach().numpy()
            # increment += len(data[j])
        else:
            # print('it did it')
            true_labels.append(all_packet_labels[j])
            data_encoded[j][:] = torch.flatten(center[-1]).detach().numpy()
            labelled_data.append(torch.flatten(center[-1]).detach().numpy())
            i+=1
            # increment += len(data[j])

    # %%

    n_cluster = n_labels  # Number of clusters
    # TODO load instead of fit.
    # Alg + Fit
    GM = GaussianMixture(n_components=n_cluster).fit(data_encoded)
    
    labelled_data = np.concatenate(
        labelled_data).reshape(-1, hidden_neurons*layers)
    for i in range(len(true_labels)):
        true_labels[i] = (true_labels[i].T / true_labels[i].sum(axis=1)).T
        true_labels[i] = true_labels[i].sum(axis=0) / len(true_labels[i])
    true_labels = np.concatenate(true_labels).reshape(-1, n_labels)
    labelled_data = labelled_data[true_labels.sum(axis=1)>0]
    true_labels = true_labels[true_labels.sum(axis=1)>0]
    # true_labels = (true_labels.T/true_labels.sum(axis=1)).T
    predicted = GM.predict(labelled_data)

    matching_matrix = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        matching_matrix[i] = np.sum(true_labels[predicted == i], axis=0)
 
    max_sum = 0
    label_index = np.arange(n_labels)
    for i in tqdm(permutations(label_index, n_cluster-1),desc='Confusion opt'):
        _sum = 0
        save_index = []
        for j in range(n_labels-1):
            _sum += matching_matrix[j, i[j]]
            save_index.append((j, i[j]))
            if j == n_labels-2:
                _sum += matching_matrix[n_labels-1, sum(label_index)-sum(i)]
                save_index.append((n_labels-1, sum(label_index)-sum(i)))
        if _sum > max_sum:
            max_sum = _sum
            combination_save = save_index

    ordered_matrix = np.zeros_like(matching_matrix)
    for i in combination_save:
        ordered_matrix.T[i[0]] = matching_matrix.T[i[1]]
    True_positives = ordered_matrix.diagonal()
    False_positives = ordered_matrix.sum(axis=1) - True_positives
    False_negatives = ordered_matrix.sum(axis=0) - True_positives
    confusion = np.array([[sum(True_positives), sum(False_positives)],
                          [sum(False_negatives), 0]])

    fig, ax = plt.subplots()
    plt.title('Confusion matrix')
    im, cbar = heatmap((ordered_matrix.T/ordered_matrix.sum(axis=1)).T,
                       label_index, label_index, ax=ax, cmap="YlGn")
    texts = annotate_heatmap(im)
    fig.tight_layout()
    plt.show()
