# -*- coding: utf-8 -*-
"""
Preprocessing module: Concatonate and group characters and words packets.

Script that loads in the preprocessed data files and group the packets in
character and keys setting up the data for training.
"""
# %%

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
# %%

if __name__ == '__main__':
    path = Path('data')
    os.chdir(path)
    for f in os.listdir():
        df = pd.read_hdf(f, key='df')
        if df.empty:
            continue
        """
        In order to associate a data points with a single STA MAC value the
        data points ta and ra are all compared to the bssid value for that
        given data point. If a ra value matches then packet was send from an
        STA to an AP and the other way around for ta values.
        The 'up' value is as such whether the STA is sender or reciever.
        If neither an exception is later made takin into consideration source
        and destination.

        Some packets have unanimouse source and destination (same sa and ta).
        If this is the case then the associated MAC address is given by the
        unanimouse source/destination belonging to the STA.
        As some transmissions occur without a sa and da, if the source and
        destination isnt unanimouse the ra and ta is utilised for association.
        """
        # Distinguish up and down for a packet through the BSSID value.
        df['source'] = df.sa[df.sa == df.ta]  # unanimouse source
        df['destination'] = df.da[df.da == df.ra]  # unanimouse destination
        # Define if a packet is up-link or down-link -- to or from an STA.
        df.loc[df.ta == df.bssid, 'up'] = False  # BSSID matches ta, STA is ra
        df.loc[df.ra == df.bssid, 'up'] = True  # BSSID matches ra, STA is ta.
        d_index = df.up.isna()  # Index value for packets with no associated UP
        # Associate each packet with a single MAC (non-ap)
        df.loc[df.up.fillna(False), 'association'] = df.loc[
            df.up.fillna(False), 'source']
        df.loc[~df.up.fillna(False), 'association'] = df.loc[
            ~df.up.fillna(False), 'destination']


        # The following is based upon empirical observations.
        # If the BSSID value doesn't indicate source and destination.
        for i in df.loc[d_index].index:
            # If source isn't unanimouse and destination is:
            if ~np.isnan(df.loc[i, 'destination'])\
                    and np.isnan(df.loc[i, 'source']):
                # Then the source is an STA
                df.loc[i, 'association'] = df.loc[i, 'ta']
                df.loc[i, 'up'] = True
            # Opposit of if statement.
            elif np.isnan(df.loc[i, 'destination'])\
                    and ~np.isnan(df.loc[i, 'source']):
                df.loc[i, 'association'] = df.loc[i, 'ra']
                df.loc[i, 'up'] = False
            else:  # Edge case where neither the above occurs -- isn't observed
                df.loc[i, 'association'] = df.loc[i, 'ta']
                df.loc[i, 'up'] = True
        df.up = df.up.astype(bool)
        """
        Group data into words and characters.

        Groups the data into "words" -- consecutive data from the same STA
        without more than 200 ms silence in between, and futher into characters
        --
        consecutive elements in a word with the same direction (up-/down-link).
        The df is required to have an association value and direction for each
        entrance (STA and direction).

        Parameters
        ----------
        Df : pandas dataframe
            The data frame containing the data.
        """
        doi = ['time', 'length', 'data_len', 'inter_arrival', 'rate', 'mcs1',
                'phy', 'totalload', 'up', 'association']
        # doi = ['length', 'data_len', 'inter_arrival', 'totalload', 'up']
        doi_bool = np.zeros(len(df.columns) + 1, dtype=bool)
        for i in range(len(df.columns)):
            doi_bool[i] = df.columns[i] in doi
        doi_bool[-1] = True

        # Grap index value for time column.
        _time_idx = df.columns.get_loc('time')
        _up_idx = df.columns.get_loc('up')  # Grap 'up' column index value.
        all_words = np.empty(df.association.unique().shape[0], dtype=object)
        i = 0
        df.rate = df.rate.fillna(-1)

        for values in tqdm(df.association.unique()):
            df.loc[df.association == values,
                   'inter_arrival'] = df.loc[df.association == values,
                                             'time'].diff(1).fillna(0)

            # STA specific df
            _STA_df = df[df.association == values].values.astype(float)
            _time_values = _STA_df.T[-1]  # Column containing inter_arrival
            # Silence between transmission break point (200 ms).
            break_point = (_time_values[1:] >= 0.2)
            # All possible index values
            _index = np.arange(0, len(_STA_df))
            # index values where the sought change occur.
            index = _index[1:][break_point]
            # array of all words
            words = np.empty(len(index) + 1, dtype=object)
            try:
                # First word is always to first break point
                words[0] = np.array(_STA_df[: index[0], doi_bool])
                # Last word is always from last break point
                words[-1] = np.array(_STA_df[index[-1]:, doi_bool])
                # Catches all the inbetween words between break points if any.
                M_words = [_STA_df[j[0]: j[1], doi_bool]
                           for j in zip(index[:-1], index[1:])]
                words[1:-1] = M_words
                del M_words
            except IndexError:
                # If only a single words exist i.e. no break points.
                words[0] = _STA_df[:, doi_bool]

            all_words[i] = words
            i += 1
        all_words = np.concatenate(all_words)
        all_words = [Tensor(word) for word in all_words]
        # for i in range(len(all_words)):
        #     all_words[i] = Tensor(all_words[i])
        # all_words = pad_sequence(all_words, batch_first=True)
        # all_words = pack_padded_sequence(all_words, enforce_sorted=False)
        # del df
        del words
        with open(f.split(sep='.')[0]+'.txt', 'wb') as fp:
            pickle.dump(all_words, fp)
