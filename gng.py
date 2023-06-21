#---------------------------------------
# -*- coding: utf-8 -*-
# Using Growing Neural Gas 1995
#---------------------------------------
import numpy as np
import math as mt
import networkx as nx
import sys
from sklearn import datasets
import matplotlib.pyplot as plt

class GNG(object):
    def __init__(self, num = 100, dim = 2, end = 100000, lam = 100, ew = 0.2, en = 0.0006, amax = 50.0, alpha = 0.5, beta = 0.9995, sig_kernel = 0.5):
        # Set Parameters

        #dimension
        self.Dim = dim
        # max of units
        self.MAX_NUM = num
        # the number of delete processes
        self.END = end
        # insert
        self.lam = lam
        # Learning coefficient
        self.Ew = ew
        # Learning coefficient of neighbors
        self.En = en
        # threshold to remove a edge
        self.AMAX = amax
        # reduction rate of Error when the insertion of a new neuron.
        self.Alpha = alpha
        # reduction rate of Error
        self.Beta = beta

        # kernel
        self.sig_kernel = sig_kernel

    def initialize_units(self):

        self.units = np.zeros((self.MAX_NUM, self.Dim))
        self.sumerror = np.zeros(self.MAX_NUM)
        self.g_units = nx.Graph()

        # reference vectors of dead units are set to infinity.
        self.units += float("inf")

        # initialize the two units
        #self.units[0], self.units[1] = data[np.random.permutation(data.shape[0])[[0, 1]]]
        self.units[0], self.units[1] = np.random.rand(self.Dim), np.random.rand(self.Dim)

        self.g_units.add_node(0)
        self.g_units.add_node(1)
        self.g_units.add_edge(0, 1, weight=0)

    def dists(self, x, units):
        #calculate distance
        return np.linalg.norm(units - x, axis=1)

    def dw(self, x, unit):
        return x - unit

    def kernel(self, x):
            return(np.exp(- np.linalg.norm(np.expand_dims(x, axis = 1) - x,axis=2)**2/2/(self.sig_kernel**2)))

    def affinity(self):
        self.units = self.units[np.isfinite(self.units[:,0])]
        A = nx.adjacency_matrix(self.g_units, weight=1)
        A = np.array(A.todense())
        A = A * self.kernel(self.units)
        return A

    def TotalError(self, data):
        total_error = 0
        for i in data:
            total_error += np.min(self.dists(i, self.units))
        return total_error/data.shape[0]

    def CountDeadUnits(self, data):
        count = np.zeros(self.MAX_NUM)
        for i in data:
            count[np.argmin(self.dists(i, self.units))] += 1
        return(np.count_nonzero(count == 0) - np.count_nonzero(self.units[:,0] == float("inf")))

    def learn(self,x, t):

        # Find the nearest and the second nearest neighbors, s_1 s_2.
        existing_units = np.array(self.g_units.nodes())
        dists = self.dists(x, self.units[existing_units])
        s_1, s_2 = existing_units[dists.argsort()[[0,1]]]

        # Add the distance between the input and the nearest neighbor s_1.
        self.sumerror[s_1] += dists[existing_units == s_1]**2

        # Move the nearest neighbor s_1 towards the input.
        self.units[s_1] += self.Ew * self.dw(x, self.units[s_1])

        if self.g_units.has_edge(s_1, s_2):
            # Set the age of the edge of s_1 and s_2 to 0.
            self.g_units[s_1][s_2]['weight'] = 0
        else:
            # Connect NN and second NN with each other.
            self.g_units.add_edge(s_1,s_2,weight = 0)

        for i in list(self.g_units.neighbors(s_1)):
            # Increase the age of all the edges emanating from the nearest neighbor s_1
            self.g_units[s_1][i]['weight'] += 1

            # Move the neighbors of s_1 towards the input.
            self.units[i] += self.En  * self.dw(x, self.units[i])

            if self.g_units[s_1][i]['weight'] > self.AMAX:
                self.g_units.remove_edge(s_1,i)
            if self.g_units.degree(i) == 0:
                self.g_units.remove_node(i)
                self.units[i] += float("inf")
                self.sumerror[i] = 0
                #end set

        # Every lambda, insert a new neuron.
        if t % self.lam == 0:
            count = 0
            nodes = list(self.g_units.nodes())

            # Find the neuron q with the maximum error.
            q = nodes[self.sumerror[nodes].argmax()]

            # Find the neighbor neuron f with maximum error.
            neighbors = list(self.g_units.neighbors(q))
            se = self.sumerror[neighbors]
            f = neighbors[np.argmax(se)]

            for i in range(self.MAX_NUM):
                if self.units[i][0] == float("inf"):
                    # Insert a new neuron r.
                    self.units[i] = 0.5 * (self.units[q] + self.units[f])
                    self.g_units.add_node(i)

                    # Insert new edges between the neuron and q and f.
                    self.g_units.add_edge(i, q, weight=0)
                    self.g_units.add_edge(i, f, weight=0)

                    # Remove the edges between q and f.
                    self.g_units.remove_edge(q, f)

                    # Decrease the error of q and f.
                    self.sumerror[q] *= self.Alpha
                    self.sumerror[f] *= self.Alpha

                    # Set the error of the new neuron to that of q
                    self.sumerror[i] = self.sumerror[q]
                    break

        # Decrease all errors.
        self.sumerror *= self.Beta


    def train(self, data):
        self.initialize_units()
        self.N = data.shape[0] # the number of data points

        for t in range(1, self.END+1):

            # Generate a random input.
            num = np.random.randint(self.N)
            x = data[num]

            self.learn(x, t)




if __name__ == '__main__':
    noisy_circles = datasets.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=0.5)

    gng = GNG(num = 100, end = 100000, lam = 250, ew = 0.1, en = 0.01, amax = 75, alpha = 0.25,  beta = 0.999, sig_kernel = 0.25)

    gng.train(noisy_circles[0])
    plt.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1])


    nx.draw_networkx_nodes(gng.g_units,gng.units,node_size=50,node_color=(0.5,1,1))
    nx.draw_networkx_edges(gng.g_units,gng.units,width=2,edge_color='b',alpha=0.5)

    plt.savefig("gng.png")

