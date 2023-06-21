#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt
import pylab as plt
import networkx as nx
from sklearn import cluster, datasets

class Kohonen:
    def __init__(self, num = 10, dim = 2, end = 2000, rate = 0.2, sigma = 1, sig_kernel = 0.5, decay = True):
        self.Dim = dim
        self.num = num**2
        self.END = end
        self.rate = rate
        self.sigma = sigma
        self.sig_kernel = sig_kernel

        # Do these parameters decay?
        self.Decay = decay

        self.g_units = nx.grid_2d_graph(num, num)
        labels = dict( ((i,j), i + (num - 1 - j) * num ) for i, j in self.g_units.nodes() )
        nx.relabel_nodes(self.g_units, labels, False)
        self.pos = np.array([[i, j] for i in range(1, num + 1) for j in range(1, num + 1)])

    def initialize_units(self):

        l = int(mt.sqrt(self.num))
        self.units = np.zeros((self.num, self.Dim))
        for i in range(self.num):
            self.units[i][0] = (i//l)/l
            self.units[i][1] = (i%l)/l

    def alpha(self, A, ac, end):
        if self.Decay == True:
            return (A * (1.0 - float(ac)/float(end + 1)) )
        else:
            return A

    def neig_func(self, dist, t):
        return np.exp(- (dist**2) / 2 / (self.alpha(self.sigma, t, self.END)**2))

    def kernel(self, x):
        return(np.exp(- np.linalg.norm(np.expand_dims(x, axis = 1) - x,axis=2)**2/2/(self.sig_kernel**2)))

    def affinity(self):
        A = nx.adjacency_matrix(self.g_units)
        A = np.array(A.todense())
        A = np.where(A > 0, 1, 0)
        A = A * self.kernel(self.units)
        return A

    def dists(self, x, units):
        #calculate distance
        return np.linalg.norm(units - x, axis=1)

    def TotalError(self, data):
        total_error = 0
        for i in data:
            total_error += np.min(self.dists(i, self.units))
        return total_error/data.shape[0]

    def CountDeadUnits(self, data):
        count = np.zeros(self.num)
        for i in data:
            count[np.argmin(self.dists(i, self.units))] += 1
        return(np.count_nonzero(count == 0))

    def learn(self, x, t):
        # Calculate distances.
        dists_x = np.linalg.norm(self.units - x, axis = 1)

        # Find the winning unit.
        min_unit_num = dists_x.argmin()

        # Adaptation.
        dists_p = np.linalg.norm(self.pos - self.pos[min_unit_num], axis=1)
        self.units +=  self.alpha(self.rate, t, self.END) * np.multiply((x - self.units), self.neig_func(dists_p, t).reshape(self.num,1))

    def train(self, data):
        for t in range(1, self.END + 1):
            x = data[np.random.choice(data.shape[0])]
            self.learn(x, t)

if __name__ == '__main__':

    n_samples = 1500
    data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    kohonen = Kohonen(num = 10, dim = data[0].shape[1], end = 100000, rate = 0.1, sigma = 0.5)
    kohonen.initialize_units(data[0])

    kohonen.train(data[0])

    plt.scatter(data[0][:,0], data[0][:,1])

    nx.draw_networkx_nodes(kohonen.g_units,kohonen.units,node_size=5,node_color=(0.5,1,1))
    nx.draw_networkx_edges(kohonen.g_units,kohonen.units,width=5,edge_color='b',alpha=0.5)
    plt.savefig(kohonen.png)
