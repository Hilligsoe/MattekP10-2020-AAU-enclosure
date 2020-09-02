#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:54:38 2020.

@author: chr
"""
import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from pytorch_modules import GRUDecoder, GRUEncoder, FullyConnected
from pytorch_modules import DatasetModule
from rough_normilisation import data_normalisation



def _decoder_step(decoder_model, hidden, xb, use_teacher_forcing=False):
    """
    Decode step with recurrent information.

    Performs the decoder step with the recurrent information feed into the
    network a vector at a time.

    Parameters
    ----------
    decoder_model : nn.module
        Pytorch decoder nn.module.
    hidden : Tensor
        Tensor containing the hidden features from previous step. The encoded
        informations compact form
    xb : list
        List containing a single batch of data from a DataLoader.
    use_teacher_forcing : bool, optional
        Bool value if teacher forcing should be used. The default is False.

    Returns
    -------
    out : Tensor
        Output predicted values.

    """
    out = []
    for j in range(len(xb)):
        dec_in = [torch.zeros(1, 8, device=device)]  # Init with null values.
        # Vector to fill the output values into.
        out_vector = torch.zeros(xb[j].shape, device=device)
        # Associated init encoded info (single vector)
        dec_hidden = hidden[:, j:j + 1, :].contiguous()
        for word in range(out_vector.shape[0]):  # For each value in the input.
            dec_in = pack_sequence(dec_in).to(device)
            out_i, dec_hidden = decoder_model(dec_in, dec_hidden)  # Decode
            dec_hidden = dec_hidden.contiguous()  # Up date the hidden states

            # Update decoder input
            dec_in = out_i.contiguous()
            if use_teacher_forcing:  # If teacher forcing use target input.
                dec_in = [xb[j][word].view(1, 8)]
            out_vector[word] = out_i.contiguous()
        out.append(out_vector)
    # Pad the output to a Tensor with correct dimension.
    out = pad_sequence(out, batch_first=True)
    return out


def loss_batch(loss_func, out_xb, yb, opt=None):
    """
    Loss function for batch size data.

    Parameters
    ----------
    loss_func : LossFunc
        Pytorch torch.nn.* loss function
    out_xb : Tensor
        Ouput data for the model given the batch of input data.
    yb : packed_tensor
        Packed tensor containing the target values.
    opt : Optimiser, optional
        Pytorch optimiser function. The default is None.

    Returns
    -------
    loss : loss
        The loss calculated from the data (gradient and value).

    """
    loss = loss_func(out_xb, yb[0])
    loss_sum = torch.sum(loss, (1, 2))
    loss = torch.sum(loss_sum / yb[1].to(device).float())
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(out_xb)


def fit_autoencoder(epochs,
                    encoder_model,
                    decoder_model,
                    center_model,
                    loss_func,
                    opt,
                    train_dl,
                    valid_dl=None,
                    validation_interval=None,
                    forced_teaching_ratio=0,
                    save_mem=True):
    """
    Fit the pytorch nn model.

    Fits the neural network model with given epochs, loss, opt and data.
    Through out the fitting, values are deleted when they stop being needed

    Parameters
    ----------
    epochs : int
        The number of epochs used in the fitting.
    encoder_model : torch.nn.modules.*
        Trainable pytorch neural network model representing encoding step.
    decoder_model : torch.nn.modules.*
        Trainable pytorch neural network model representing decoding step.
    center_model : torch.nn.modules.*
        Trainable pytorch neural network model representing center step of
        autoencoder.
    loss_func : LossFunc
        Differentiable loss function of the neural network.
    opt : Optimiser
        Pytorch optimiser function.
    train_dl : DataLoader
        DataLoader associated with the training dataset.
    valid_dl : DataLoader
        DataLoader associated with the training dataset.
    validation_interval : int
        How often the validation data is tested during training, nr of epochs
        since last test. The Default is None.
    forced_teaching_ratio : int
        Probabilistic ratio for when teacher forcing should be utilised.
        The Default is 0.
    save_mem : bool
        Indicate if the teacher forcing should utilise a memory saving
        algorithm or a significantly faster but memory consuming algorithm.
        The Default is True.

    Returns
    -------
    saved_loss : array_like
        Array containing the loss for plotting.

    """
    saved_loss = np.zeros(epochs)
    try:
        saved_val_loss = np.zeros(int(epochs/validation_interval))
    except TypeError:
        saved_val_loss = 0
    for epoch in range(epochs):
        encoder_model.train()
        center_model.train()
        decoder_model.train()
        _temp = []

        for xb in tqdm(train_dl, desc=f'Epoch:{epoch}'):
            # TODO Create yb and x_target from xb.
            x_input = pack_sequence(xb, enforce_sorted=False).to(device)
            hidden = encoder_model(x_input).to(device)
            center = center_model(hidden).contiguous().to(device)
            input_unpacked = pad_packed_sequence(x_input, batch_first=True)

            use_teacher_forcing = True if\
                np.random.random() > forced_teaching_ratio else False

            # Low mem -- slow.
            if save_mem:
                out = _decoder_step(decoder_model, center, xb,
                                    use_teacher_forcing=use_teacher_forcing)
            # Teacher forcing high mem -- Fast, dupes data.
            else:
                x_target = [torch.cat(
                                (torch.zeros((1, 8), device=device),
                                 xb[i][:-1].to(device)), 0)
                            for i in range(len(xb))]
                x_target = pack_sequence(x_target,
                                         enforce_sorted=False).to(device)
                out, dec_hidden = decoder_model(x_target, center)
                # Delete variable only present if mem isn't saved
                del x_target
                del dec_hidden

            # Calculate loss
            _temp.append(loss_batch(loss_func, out, input_unpacked, opt))
            # Delete loop dependent variable
            del x_input
            del out

        losses, nums = zip(*_temp)
        saved_loss[epoch] = float(np.sum(np.multiply(losses, nums))
                                  / np.sum(nums))
        # Delete out of loop dependent variable
        del xb
        del hidden
        del center
        del input_unpacked

        # Validation fo the model if set gvein
        if valid_dl is not None and validation_interval is not None:
            if (epoch+1) % validation_interval == 0:
                encoder_model.eval()
                center_model.eval()
                decoder_model.eval()
                _temp = []
                for vb in tqdm(valid_dl, desc=f'Valid:{epoch}'):
                    with torch.no_grad():
                        v_input = pack_sequence(
                            vb, enforce_sorted=False).to(device)

                        hidden = encoder_model(v_input).to(device)
                        center = center_model(hidden).contiguous().to(
                                                                    device)
                        input_unpacked = pad_packed_sequence(
                                        v_input, batch_first=True)
                        out = _decoder_step(decoder_model, center, vb,
                                            use_teacher_forcing=True)

                    _temp.append(
                        loss_batch(loss_func, out, input_unpacked)
                                 )
                losses, nums = zip(*_temp)
                val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                saved_val_loss[int(epoch/validation_interval)] = val_loss
            del _temp
    return saved_loss, saved_val_loss


def collate_fn_autoencoder(_list):
    """
    Colate function for the DataLoader.

    Function needed to pass the list of tensors into list of batch size.

    Parameters
    ----------
    _list : list
        input list.

    Returns
    -------
    _list : list
        Sorted input list.

    """
    _list.sort(key=len, reverse=True)
    return _list

# SOS and EOS: Start/End of Sequence/string.


if __name__ == '__main__':
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)  # Fixed seed for numpy (identical shuffle)
    hidden_neurons = 140 
    layers = 2
    epochs = 150
    validation_ratio = 1/8
    validation_interval = int(validation_ratio*epochs)
    data = []
    path = Path('/media/cht/Local Disk//autoencoder_data')
    os.chdir(path)

    for f in os.listdir():
        with open(f, 'rb') as fp:
            all_words = pickle.load(fp)
        data += all_words
    data = data_normalisation(data)
    valid_part = 0.3
    np.random.shuffle(data)
    train = data[: -int(len(data)*valid_part)]
    valid = data[-int(len(data)*valid_part):]
    del data
    # Composit the autoencoder from its independent models
    encoder_model = GRUEncoder(hidden_neurons, layers)
    center_model = FullyConnected(hidden_neurons).to(device)
    decoder_model = GRUDecoder(hidden_neurons, layers).to(device)
    encoder_model.to(device)
    # Composition of the autoencoders parameters
    model_parameters = list(encoder_model.parameters())\
        + list(center_model.parameters()) + list(decoder_model.parameters())

    loss_func = nn.L1Loss(reduction='none')
    optimiser = Adam(model_parameters,  lr=0.001)

    batch_size = 32
    train_ds = DatasetModule(train)
    valid_ds = DatasetModule(valid)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=collate_fn_autoencoder)

    valid_loader = DataLoader(valid_ds, batch_size=batch_size*4,
                              shuffle=True, num_workers=0,
                              collate_fn=collate_fn_autoencoder)
# %%
    loss, v_loss = fit_autoencoder(epochs, encoder_model, decoder_model,
                                   center_model, loss_func, optimiser,
                                   train_loader,
                                   valid_dl=valid_loader,
                                   validation_interval=validation_interval,
                                   forced_teaching_ratio=1,
                                   save_mem=False)
