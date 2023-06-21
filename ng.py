#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt
import networkx as nx
import sys
from scipy import ndimage
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.stats import rankdata

class NG(object):
    def __init__(self, num = 100, dim = 2, end = 1000000, lam_i = 30.0, lam_f = 0.01, ew_i = 0.3, ew_f = 0.05, amax_i = 20.0, amax_f = 200, sig_kernel = 0.5, decay = True):
        # Set Parameters

        #dimension
        self.Dim = dim

        # max of units
        self.NUM = num
        # relationship of neighbors
        self.lam_i = lam_i
        self.lam_f = lam_f
        # Learning coefficient
        self.Ew_i = ew_i
        self.Ew_f = ew_f
        # threshold to remove a edge (lifetime of edge T)
        self.AMAX_i = amax_i
        self.AMAX_f = amax_f

        # Stopping condision
        self.END = end

        #kernel
        self.sig_kernel = sig_kernel

        #Do these parameters decay?
        self.Decay = decay

    def initialize_units(self):

        self.g_units = nx.Graph()

        # initialize the units
        #self.units = data[np.random.permutation(self.N)[range(self.NUM)]]
        self.units = np.random.rand(self.NUM, self.Dim)

        for i in range(self.NUM):
            self.g_units.add_node(i)

    def dists(self, x, units):
        #calculate distance
        return np.linalg.norm(units - x, axis=1)

    def dw(self, x, unit):
        return x - unit

    def TotalError(self, data):
        total_error = 0
        for i in data:
            total_error += np.min(self.dists(i, self.units))
        return total_error/data.shape[0]

    def CountDeadUnits(self, data):
        count = np.zeros(self.NUM)
        for i in data:
            count[np.argmin(self.dists(i, self.units))] += 1
        return(np.count_nonzero(count == 0))

    def kernel(self, x):
        return(np.exp(- np.linalg.norm(np.expand_dims(x, axis = 1) - x,axis=2)**2/2/(self.sig_kernel**2)))

    def affinity(self):
        A = nx.adjacency_matrix(self.g_units)
        A = np.array(A.todense())
        A = np.where(A > 0, 1, 0)
        A = A * self.kernel(self.units)
        return A

    def normalize(self, data):
    # normalize dataset
        self.mindata = data[np.argmin(np.linalg.norm(data, axis=1))]
        self.diff_max_min = np.linalg.norm( data[np.argmax(np.linalg.norm(data, axis=1))] - data[np.argmin(np.linalg.norm(data, axis=1))])
        data = (data - self.mindata) / self.diff_max_min
        return data

    def gt(self, gi, gf, t, tmax):
        if self.Decay == True:
            return gi * ( (gf / gi) ** (t/tmax) )
        else:
            return gi

    def learn(self, x, t):
        # Find the nearest and the second nearest neighbors, s_1 s_2.
        dists = self.dists(x, self.units)
        sequence = dists.argsort()

        # Move the neurons towards the input.
        self.units += self.gt(self.Ew_i, self.Ew_f, t, self.END) * np.expand_dims(np.exp(- (rankdata(dists) - 1) / self.gt(self.lam_i, self.lam_f, t, self.END)), axis = 1) * self.dw(x, self.units)

        n_1, n_2 = sequence[[0,1]]
        if self.g_units.has_edge(n_1, n_2):
            # Set the age of the edge of the nearest neighbor and the second nearest neighbor to 0.
            self.g_units[n_1][n_2]['weight'] = 0
        else:
            # Connect the nearest neighbor and the second nearest neighbor with each other.
            self.g_units.add_edge(n_1,n_2,weight = 0)

        for i in list(self.g_units.neighbors(n_1)):
            # Increase the age of all the edges emanating from the nearest neighbor
            self.g_units[n_1][i]['weight'] += 1

            # remove the edge of the nearest neighbor with age > lifetime
            if self.g_units[n_1][i]['weight'] > self.gt(self.AMAX_i, self.AMAX_f, t, self.END):
                #if self.g_units.degree(n_1) > 1 and self.g_units.degree(i) > 1: # If you permit a node with no edge, commentout this line.
                self.g_units.remove_edge(n_1,i)

    def train(self, data):
        self.initialize_units()

        self.N = data.shape[0] # the number of data points

        for t in range(self.END):
            # Generate a random input.
            num = np.random.randint(self.N)
            x = data[num]
            self.learn(x, t)

if __name__ == '__main__':

    data = datasets.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=0.5)

    ng = NG(num = 100, end = 100000, lam_i = 5.0, lam_f = 0.01, ew_i = 0.1, ew_f = 0.05, amax_i = 40.0, amax_f = 400.0, sig_kernel = 0.5)
    ng.train(data[0])

    plt.scatter(data[0][:,0], data[0][:,1])
    nx.draw_networkx_nodes(ng.g_units,ng.units,node_size=5,node_color=(0.5,1,1))
    nx.draw_networkx_edges(ng.g_units,ng.units,width=2,edge_color='b',alpha=0.5)

    plt.savefig("ng.png")
