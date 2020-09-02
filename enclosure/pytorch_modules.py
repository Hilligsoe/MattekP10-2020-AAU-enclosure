# -*- coding: utf-8 -*-
"""
Pytorch: Setup data in tensors for training of RNN.

Script that loads the grouped data and set it up in tensors (Packing and
padding), as well as split it for training and testing purpose.
"""

import torch.nn as nn
from torch import transpose
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_sequence
# from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# test = [[Tensor(i) for i in j] for j in all_words]
# a = [pack_sequence(i, enforce_sorted=False) for i in test]
# # pad_sequence instead?

# =============================================================================
# Encoder module
# =============================================================================


class GRUEncoder(nn.Module):
    """
    GatedRecurrentUnit Encoder: Encoder part of Autoencoder - Costum nn module.

    Custom pytorch nn module -- GRU encoder class for ease of setup.
    The class contains the necessary setup in order to performe encoding step.
    For GRU documentation refer to the pytorch docs for GRU and Custom nn.

    https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
    https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html

    Parameters
    ----------
    hidden_size : int
        Expected size of the hidden layer.
    num_layers : int
        Number of recurrent layers
    input_size : int, optional
        Size of the input. Default is 8.

    Input
    -----
    x : Tensor

    Returns
    -------
    hidden : Tensor
    """

    def __init__(self, hidden_size, num_layers, input_size=8):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.GRU(input_size, hidden_size, num_layers)
        self.drop = nn.Dropout(p=0.2)  # 20% dropout

    def encode(self, x):
        """
        Encode via the GRUEncoder.

        The documentation can stil refer to the pytorch docs nn.GRU.

        Parameters
        ----------
        x : Tensor
            Tensor containing the input features.

        Returns
        -------
        hidden : Dataset
            Tensor containing the output features (hidden values).

        """
        _, hidden = self.encoder(x)
        hidden = self.drop(hidden)
        return hidden

    def forward(self, x):
        """
        Forward pass of custom neural network module.

        Tensor containing the features that should be passed forwards in the
        network -- required for use as a costum NN.

        Parameters
        ----------
        x : Tensor
            Tensor containing the input features -- possibly from previus NN
            layer.

        Returns
        -------
        hidden : Tensor
            Tensor containing the output features passed forward in the NN.

        """
        hidden = self.encode(x)
        return hidden

# =============================================================================
# Decoder module
# =============================================================================


class GRUDecoder(nn.Module):
    """
    GatedRecurrentUnit Decoder: Decoder part of Autoencoder - Costum nn module.

    Custome pytorch nn module -- GRU Decoder class.
    The class contains the necessary setup in order to performe the decoding
    step.
    For GRU documentation refer to the pytorch docs for GRU and Custom nn.

    https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
    https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
    """

    def __init__(self, hidden_size, num_layers, input_size=8):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder = nn.GRU(input_size, hidden_size, num_layers)
        self.lin = nn.Linear(hidden_size, input_size)  # Linear transform

    def decode(self, x, hidden):
        """
        Decode via the GRUDecoder.

        Beside the decoding (Utilisation of NN se pytorch doc), data this is
        setup to handle both training and encoding, as such it takes input
        values of both hidden and data.

        Parameters
        ----------
        x : Tensor
            Tensor containing the features for decoding.
        hidden : Tensor
            Tensor containing the output features h_t from the last layer of
            the GRU, for each t.

        Returns
        -------
        decoder_output : Tensor
            Tensor containing output for the decoder state -- this is prossed
            such that the value is as the prior to encoding.
        hidden : Tensor
            Tensor containing out for the decoder state.

        """
        decoder_output, hidden = self.decoder(x, hidden)
        decoder_output = pad_packed_sequence(decoder_output,
                                             batch_first=True)[0]
        decoder_output = transpose(decoder_output, 0, 1)
        decoder_output = self.lin(decoder_output)
        decoder_output = transpose(decoder_output, 0, 1)
        return decoder_output, hidden

    def forward(self, x, hidden):
        """
        Forward pass of custome neural network module.

        Parameters
        ----------
        x : Tensor
            Tensor containing the input fetures for the neural network.
        hidden : Tensor
            Tensor containing the initial hidden state for each element in
             the batch.

        Returns
        -------
        output : Tensor
            Tensor containing the output features h_t from the last layer of
            the GRU, for each t.
        hidden : Tensor
            Tensor containing the hidden state for t = seq_len

        """
        output, hidden = self.decode(x, hidden)
        return output, hidden

# =============================================================================
# Fully connected module
# =============================================================================


class FullyConnected(nn.Module):
    """
    Fully connected layer -- center of Autoencoder.

    Custome pytorch nn module -- Fully connected layer.
    The class contains the necessary setup in order to transform the values of
    the encoder into suitable value for input in the decoder -- encoder end
    step.
    For GRU documentation refer to the pytorch docs for GRU and Custom nn.

    Pytorch handle a "standard" neural network layer in two steås, by firstly
    performing the linear transformation and and then passing it through the
    non-linear activation function. E.g. Tanh in this case.

    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html
    """

    def __init__(self, hidden_size):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)  # Input/output size
        self.tan = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of costume neural network module.

        Parameters
        ----------
        x : Tensor
            Tensor containing the input features.

        Returns
        -------
        connect : Tensor
            Tensor containing the output feature -- tanh(linear(x)).
        """
        x = transpose(x, 0, 1)
        connect = self.fc(x)
        connect = self.tan(connect)
        connect = transpose(connect, 0, 1)
        return connect

# =============================================================================
# Pytorch DatasetModule
# =============================================================================


class DatasetModule(Dataset):
    """
    Map-style dataset.

    Implements a the magic methods: __len__ and __getitem__ and represents a
    map from (possibly non-integral) indices/keys to data samples.
    https://pytorch.org/docs/stable/data.html
    """

    def __init__(self, input_list):
        self.sentences = input_list

    def __len__(self):
        """
        Dunder length option internally used for pytorch datasets.

        Returns
        -------
        length : int
            The length of the dataset.

        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Dunder get item internally used for pytorch datasets.

        Given an index value grap the associated sample / data value.

        Parameters
        ----------
        idx : int
            Index value.

        Returns
        -------
        sample : Tensor
            Tensor value associated with the index.

        """
        sample = self.sentences[idx]
        return sample
