# -*- coding: utf-8 -*-
"""
Preprocessing module: Load and manipulate pcapng data for fingerprinting.

Script that loads in a captured .pcapng file converted to a .csv file.
Using numpy and pandas the data is saved from a data frame to .hdf5 format.
"""
# %%

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
# %%


class data_preprocessing():
    """Perform preprocessing on .pcapng data and setup for fingerprinting."""

    def __init__(self, data_directory='../data'):
        self._read_csv(data_directory)

    def _read_csv(self, data_directory):
        """
        Read and manipulat a csv file using pandas for fingerprinting.

        Loads in the csv files consecutively and calculate/streamline the data.
        Removes non data data-points and averages the lost information
        (airtime - total load).
        The dtype of the data is changed to a suitible value and the data is
        transformed in order to minimise the required memory.
        All nan values are filled with either a -1 if the data is catagorical
        or a 0 if its continuous. NaN values are only present in the data if
        the associated packet doesn't have the given propertie.
        The total load calculation is the sum of all trafik load on the network
        during the last 200 ms.

        Parameters
        ----------
        data_directory : str
            Path to data directory.

        Hidden Parameters
        -----------------
        _name : array_like
            List of names associated with the data corresponding to tshark csv
            conversion.
        _bool_a : array_like
            boolean array picking out specific columns in a pandas data frame.
        """
        _dic = Path(data_directory)  # Data directory path
        # Names for columns index.
        _name = ['time', 'time_d', 'mactime', 'length', 'sa', 'da', 'ta', 'ra',
                 'bssid', 'staa', 'dbm_antsignal', 'b_checksum', 'fc_type',
                 'data_len', 'phy', 'rate', 'mcs1', 'mcs2', 'mcs3', 'mcs4',
                 'mcs5']
        # Some columns in the panda file can is ignored due to deprication.
        _bool_a = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0], dtype=bool)
        try:
            encoded = np.load('mac_encoded.npy', allow_pickle=True)
        except FileNotFoundError:
            pass
        enc = OrdinalEncoder()
        u_mac = np.array([])  # array with uniq mac values for encoding
        # Encode mac value identically for all data points accross all files.
        for f in os.listdir(_dic):
            df = pd.read_csv(_dic/f, sep=';', names=_name, low_memory=False)
            u_mac = np.hstack((u_mac,
                               np.unique(df.loc[:, 'sa':'bssid'].fillna(
                                                         'nan').values.ravel())
                               ))
        u_mac = np.unique(u_mac)
        u_mac = np.unique(np.hstack((encoded[:, 0], u_mac)))
        # Save a copy of the encoding and MAC address.
        mac_encoded = np.hstack(
                            (u_mac.reshape(-1, 1),
                             enc.fit_transform(u_mac.reshape(-1, 1)))
                            )
        np.save('mac_encoded', mac_encoded)

        for f in os.listdir(_dic):
            done = [i.split(sep='.')[0]+'.csv' for i in os.listdir('data')]
            # TODO Consider the utilisation of multiprocessing.
            if f in done:
                continue
            else:
                pass
            df = pd.read_csv(_dic/f, sep=';', names=_name, low_memory=False)
            df = df.loc[:, _bool_a]

            # Calculate total load.
            k = 0  # Initial index value for optimised search.
            df['totalload'] = -1  # Initial total load value -- Depricated
            # wlan.fc.type = 2 indication for a data transmission.
            for i in tqdm(np.argwhere(df.fc_type.values == 2), desc=f):
                # Only search form the last 200 ms mark to the current value.
                try:
                    k = np.argwhere(
                        (df.time.values[k[0]:i]
                         - df.time.values[i] + 0.2)[k[0]:i] >= 0)[0]
                except (IndexError, TypeError):
                    k = np.argwhere(
                        (df.time.values - df.time.values[i] + 0.2) >= 0)[0]
                df.loc[i, 'totalload'] = np.sum(df.length.values[k[0]:i[0]],
                                                axis=0)

            # Remove all packets not containing data or incorrectly received
            df['fc_type'] = df['fc_type'].fillna(-1).astype(np.int8)
            df['b_checksum'] = df['b_checksum'].fillna(0).astype(bool)
            df = df[df.fc_type == 2]  # Take only the packets containing data.
            df = df[~df.b_checksum]  # Only correcly received packets

            # Format data (dtype) for storage and ease.
            df['length'] = df['length'].astype(np.int32)
            # Catch incorrectly given data_len
            try:
                df['data_len'] = df['data_len'].fillna(0).astype(np.int32)
            except ValueError:
                df['data_len'] = df['data_len'].fillna(0).astype(str)
                for i in df.index:
                    df.loc[i, 'data_len'] = df.loc['data_len',
                                                   i].split(sep=',')[0]

                df['data_len'] = df['data_len'].astype(np.int32)

            df['phy'] = df['phy'].fillna(-1).astype(np.int8)
            df['mcs1'] = df['mcs1'].fillna(-1).astype(np.int8)
            try:
                for mac_type in _name[4:9]:
                    df[mac_type] = enc.transform(df[mac_type].fillna(
                                    'nan').values.reshape(-1, 1)
                                                 ).astype(np.int32)
            except ValueError:
                pass

            save_path = Path('data/'+f.split(sep='.')[0]+'.h5')
            # print(f'Saving: {f.split(sep=".")[0]}')
            df.to_hdf(save_path, key='df')
# %%


if __name__ == '__main__':
    # Names of data categories in order of aaperance in CSV file
    data_preprocessing('data')
