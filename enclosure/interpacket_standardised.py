#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 13:06:55 2020

@author: cht
"""

import os
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from label_dict import label_dict
from bin_label_dict import bin_label_dict
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

if __name__ == "__main__":
    data = []
    mac_encoded = [81343, 82447, 66391, 67314]
    all_packet_labels = []
    
    n_labels = 5
    if n_labels == 2:
        label_dict = bin_label_dict

    for f in os.listdir('/data/'):
        with open('f, 'rb') as fp:
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
            

    # data = data_normalisation(data, association=True)
    to_pd = np.concatenate(data)
    
    doi = ['time', 'length', 'data_length', 'phy', 'rate', 'msc1', 'totalload',
           'up', 'association', 'inter_arrival']
    doi_bool = np.ones_like(doi, dtype=bool)
    doi_bool[0] = False
    doi_bool[-2] = False
    df = pd.DataFrame(to_pd[:])
    df.columns = doi
    df = df.T[doi_bool].T

    interpacket = np.vstack((df.length, df.inter_arrival)).T
    n_cluster = n_labels  # Number of clusters
    # kmeans = KMeans(n_clusters=n_cluster,
    #                 random_state=0).fit(interpacket)
    GM = GaussianMixture(n_components=n_cluster, verbose=2).fit(interpacket)
    labelled_data = []
    true_labels = []
    for i in range(len(data)):
            if len(all_packet_labels[i].shape) == 1:
                pass
            else:
                labelled_data.append(data[i])
                true_labels.append(all_packet_labels[i])

    labelled_data = np.concatenate(labelled_data)
    labelled_data = np.vstack((labelled_data.T[1], labelled_data.T[-1])).T
    true_labels = np.concatenate(true_labels)
    labelled_data = labelled_data[true_labels.sum(axis=1)>0]
    true_labels = true_labels[true_labels.sum(axis=1)>0]
    true_labels = (true_labels.T/true_labels.sum(axis=1)).T
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
