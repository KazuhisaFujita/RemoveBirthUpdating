#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt
import pylab as pl
import networkx as nx
import sys
from scipy import ndimage
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
from scipy.stats import rankdata

class NGRB(object):
    def __init__(self, num = 100, dim = 2, end = 10000, lam = 1.0, ew = 0.2, amax = 100.0, sig_kernel = 0.5, rb_metric = "num_wins", th_rb = 0.01, beta = 0.0005):
        # Set Parameters

        #dimension
        self.Dim = dim
        # max of units
        self.NUM = num
        # relationship of neighbors
        self.lam = lam
        # Learning coefficient
        self.Ew = ew
        # threshold to remove a edge (lifetime of edge T)
        self.AMAX = amax

        # Stopping condision
        self.END = end

        # Metric for RB updating
        self.Metric = rb_metric
        # threshold for RB processc
        self.Th_RB = th_rb
        # decay of error
        self.Beta = beta

        #kernel
        self.sig_kernel = sig_kernel

        #RB updating counter
        self.counter_rb = 0

    def initialize_units(self):

        self.g_units = nx.Graph()

        # initialize the units
        #self.units = data[np.random.permutation(self.N)[range(self.NUM)]]
        self.units = np.random.rand(self.NUM, self.Dim)

        self.wins = np.zeros(self.NUM)
        self.E = np.zeros(self.NUM)
        self.U = np.zeros(self.NUM)

        # count of units
        self.u_count = np.zeros(self.NUM)

        for i in range(self.NUM):
            self.g_units.add_node(i)

    def dists(self, x, units):
        #calculate distance
        return np.linalg.norm(units - x, axis=1)

    def dw(self, x, unit):
        return x - unit

    def kernel(self, x):
        return(np.exp(- np.linalg.norm(np.expand_dims(x, axis = 1) - x,axis=2)**2/2/(self.sig_kernel**2)))

    def affinity(self):
        A = nx.adjacency_matrix(self.g_units)
        A = np.array(A.todense())
        A = np.where(A > 0, 1, 0)
        A = A * self.kernel(self.units)
        return A

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

    def RemoveCreateUnit(self, min_u, max_u):
        self.counter_rb += 1

        # Remove the unit with minimum wins.
        self.g_units.remove_node(min_u)

        # List the neighbors of the unit with most wins
        neighbors = list(self.g_units.neighbors(max_u))

        if self.Metric == "num_wins":
            se = self.wins[neighbors]
        elif self.Metric == "error":
            se = self.E[neighbors]

        if len(neighbors) != 0:
            # Find the neighboring unit with most wins.
            f = neighbors[np.argmax(se)]
        else:
            # Find the nearest unit of the unit with most wins.
            f = np.linalg.norm(self.units - self.units[max_u], axis = 1).argsort()[1]

        self.units[min_u] = (self.units[max_u] + self.units[f])/2
        self.wins[min_u]  = (self.wins[max_u]  + self.wins[f])/2
        self.E[min_u]     = (self.E[max_u]     + self.E[f])/2
        self.U[min_u]     = (self.U[max_u]     + self.U[f])/2

        self.g_units.add_node(min_u)
        self.g_units.add_edge(min_u, max_u, weight = 0)
        self.g_units.add_edge(min_u, f, weight = 0)

    def RBupdating(self, dists_x):
        # Remove and birth process

        if self.Metric == "num_wins":
            n_win = dists_x.argmin()

            self.wins[n_win] += 1

            # the unit with the maximum number of winnig
            max_u = np.argmax(self.wins)
            # the unit with the minimum number of winnig
            min_u = np.argmin(self.wins)

            if (self.wins[min_u]/self.wins[max_u] < self.Th_RB):
                # Put the unit with the minimum number of winnig around the unit with maximum number of winning.
                self.RemoveCreateUnit(min_u, max_u)

            # Decrease all errors and utility measure.
            self.wins -= self.Beta * self.wins

        elif self.Metric == "error":
            n_win, n_win2 = dists_x.argsort()[[0, 1]]

            self.E[n_win] += dists_x[n_win]**2
            self.U[n_win] += dists_x[n_win2]**2 - dists_x[n_win]**2

            min_u = np.argmin(self.U)
            max_u = np.argmax(self.E)
            if (self.U[min_u] / self.E[max_u] < self.Th_RB) and min_u != max_u:
                self.RemoveCreateUnit(min_u, max_u)

            # Decrease all errors and utility measure.
            self.E -= self.Beta * self.E
            self.U -= self.Beta * self.U

    def learn(self, x, t):

        # Find the nearest and the second nearest neighbors, s_1 s_2.
        dists = self.dists(x, self.units)

        # Sort the distances between the input data point and the units.
        sequence = dists.argsort()

        #Move the neurons towards the input.
        self.units += self.Ew * np.expand_dims(np.exp(- rankdata(dists) / self.lam), axis = 1) * self.dw(x, self.units)

        # Find the winning and the second winning units.
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
            if self.g_units[n_1][i]['weight'] > self.AMAX:
                self.g_units.remove_edge(n_1,i)

        #RB updating
        self.RBupdating(dists)

    def train(self, data):
        self.N = data.shape[0] # the number of data points
        self.initialize_units()

        for t in range(1, self.END+1):
            # Generate a random input.
            num = np.random.randint(self.N)
            x = data[num]

            self.learn(x, t)

if __name__ == '__main__':

    noisy_circles = datasets.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=0.5)

    ng = NGRB(num = 100, end = 100000)
    ng.train(noisy_circles[0])

    plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])
    nx.draw_networkx_nodes(ng.g_units,ng.units,node_size=5,node_color=(0.5,1,1))
    nx.draw_networkx_edges(ng.g_units,ng.units,width=2,edge_color='b',alpha=0.5)
    plt.savefig("ngrb.png")
