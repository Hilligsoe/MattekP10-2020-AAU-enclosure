# -*- coding: utf-8 -*-
"""
Script that loads in a captured .pcapng file converted to a .csv file.
Using numpy pandas and matplotlib, the data is stored in dataframes and
aspects visulised.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Names of data categories in order of aaperance in CSV file

names = ['Time', 'Time_delta', 'Length', 'MAC_sa', 'MAC_da', 'MAC_ta',
         'MAC_ra', 'STA', 'BSSID', 'Signal_dB', 'RSSI', 'Datarate',
         'wlan_flags', 'radio_flags', 'type', 'subtype',
         'Phy', 'sa_resolved', 'da_resolved', 'bad_checksum']


ifname = '*'  # Input FileName
file = ifname+'.csv'  # Name of csv file
sdic = 'data_images/'
color = 'C0'

"""
# Panda read CSV file
"""
df = pd.read_csv(file, sep=';', names=names)

# Different values that is used.
N = df.shape[0]  # Number for data points
checksum = np.invert(df.bad_checksum.fillna(0).values.astype(bool))  # Bad data
interest = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0], dtype=bool)  # Data of interest, disused.
t2 = (df.type == 2).values
#t2 = True
STAs = df.STA[checksum].fillna('NaN')
u_sta = np.unique(STAs)
unique_sta, sta_counts = np.unique(STAs, return_counts=True)
sa = np.unique(df.MAC_sa[checksum].fillna('NaN'))

"""
Elements for analysis
"""
# MofI: Mac of Interest
MofI = 'ff:ff:ff:ff:ff:ff'

boolian = ((df.MAC_da == MofI) | (df.MAC_ra == MofI)).values * checksum * t2
# time = df.MAC_time[boolian].values  # Mac time vs normalised time.
time = df.Time[boolian].values
tau = time[1:] - time[:-1]  # Interpacket time (Dif in time between 2 packets)
length = df.Length[boolian].values[:-1]
MA_size = 10
MA_ = np.convolve((length/tau), np.ones((MA_size,))/MA_size, mode='valid')
# tau = tau/np.max(tau)  # Tested normalisation

"""
Visulaisation for the two cases including both source and destination.
"""
# Interpacket arrival, is there a combination of packet length and time for
# specific applications
# Destination
plt.figure('Inter_Packet_Arrival_Destination')
plt.title('Inter Packet Arrival -- Destination')
plt.plot(tau, length, '.', color=color, label=r'$\tau_i, L_i$')
plt.ylabel(r'Packet length $L_i$')
plt.xlabel(r'Distance between packets in time $\tau_i$')
plt.legend()
plt.show()
#plt.savefig(sdic+ifname+'Inter_Packet_Arrival_Destination.pdf')
#plt.savefig(sdic+ifname+'Inter_Packet_Arrival_Destination.png')

# This one is the boring one and shouldnt give much information
plt.figure('Packet_Arrival_Destination')
plt.title('Packet Arrival -- Destination')
plt.plot(time[:-1], length, '.', color=color, label=r'$\tau_i, L_i$')
plt.ylabel(r'Packet length $L_i$')
plt.xlabel(r'Packets arrival in time $\tau_i$')
plt.legend()
plt.show()
#plt.savefig(sdic+ifname+'Packet_Arrival_Destination.pdf')
#plt.savefig(sdic+ifname+'Packet_Arrival_Destination.png')

plt.figure('packet_average_throughput_est')
plt.title('MA Length/Tau')
plt.plot(time[MA_size:], MA_, '.')
plt.xlabel('Sample number')
plt.ylabel('Calculated average')
plt.show()

# Source
boolian = ((df.MAC_sa == MofI) | (df.MAC_ta == MofI)).values * checksum * t2
#time = df.MAC_time[boolian].values
time_s = df.Time[boolian].values
tau_s = time_s[1:] - time_s[:-1]
length_s = df.Length[boolian].values[:-1]
#tau = tau/np.max(tau)
# Interpacket arrival, is there a combination of packet length and time for
# specific applications.
plt.figure('Inter_Packet_Arrival_Source')
plt.title('Inter Packet Arrival -- Source')
plt.plot(tau_s, length_s, '.', color=color, label=r'$\tau_i, L_i$')
plt.ylabel(r'Packet length $L_i$')
plt.xlabel(r'Distance between packets in time $\tau_i$')
plt.legend()
plt.show()
#plt.savefig(sdic+ifname+'Inter_Packet_Arrival_Source.pdf')
#plt.savefig(sdic+ifname+'Inter_Packet_Arrival_Source.png')

# This one is the boring one and shouldnt give much information
plt.figure('Packet_Arrival_Source')
plt.title('Packet Arrival -- Source')
plt.plot(time_s[:-1], length_s, '.', color=color, label=r'$\tau_i, L_i$')
plt.ylabel(r'Packet length $L_i$')
plt.xlabel(r'Packets arrival in time $\tau_i$')
plt.legend()
plt.show()
#plt.savefig(sdic+ifname+'Packet_Arrival_Source.pdf')
#plt.savefig(sdic+ifname+'Packet_Arrival_Source.png')

"""
Clustering
"""
X = np.vstack((tau[MA_size:], length[MA_size:])).T
pca = PCA(n_components=2)  # Evt. try MLE
Y = pca.fit_transform(X)
plt.figure('quick')
plt.plot(Y.T[1], Y.T[0], '.', color=color)
plt.show()
# Data reformat for sklearns KMeans.
n_cluster = 4  # Number of clusters
kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)  # Alg + Fit
labels = kmeans.labels_  # Labels produced by alg

# Visualisation of clusters
plt.figure('clusters')
plt.title('Clustering')
for i in range(n_cluster):
    plt.plot(time[MA_size:-1][labels == i], length[MA_size:][labels == i], '.')
plt.ylabel(r'Packet length $L_i$')
plt.xlabel(r'Packets arrival in time $\tau_i$')
plt.show()

plt.figure('clusters_inter')
plt.title('Clustering Interpacket')
for i in range(n_cluster):
    plt.plot(tau[MA_size:][labels == i], length[MA_size:][labels == i], '.')
plt.ylabel(r'Packet length $L_i$')
plt.xlabel(r'Packets arrival in time $\tau_i$')
plt.show()
