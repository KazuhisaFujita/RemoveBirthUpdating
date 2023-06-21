#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt
import networkx as nx
import sys
from sklearn import datasets
import matplotlib.pyplot as plt


class OKMEANS():
    def __init__(self, num = 100, dim = 2, lam = 0.1, end = 100000, decay = True):

        #dimension
        self.Dim = dim
        # Set Parameters
        self.lam = lam
        # max of units
        self.NUM = num
        # the number of delete processes
        self.END = end

        # Do the parameter decay?
        self.Decay = decay

    def initialize_units(self):
        #self.units = data[np.random.permutation(self.N)[range(self.MAX_NUM)]]
        self.units = np.random.rand(self.NUM, self.Dim)

    def dists(self, x, units):
        #calculate distance
        return np.linalg.norm(units - x, axis=1)

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

    def alpha(self, A, ac, end):
        if self.Decay == True:
            return (A * (1.0 - float(ac)/float(end + 1)) )
        else:
            return A

    def learn(self, x, t):
        # Calculate distances between x and reference vectors.
        dists_x = np.linalg.norm(self.units - x, axis = 1)

        # Find a winning unit.
        min_unit_num = dists_x.argmin()

        # Adaptation
        self.units[min_unit_num] +=  self.alpha(self.lam, t, self.END) * (x - self.units[min_unit_num])

    def train(self, data):
        self.initialize_units()

        for t in range(1, self.END + 1):
            x = data[np.random.choice(data.shape[0])]
            self.learn(x, t)

if __name__ == '__main__':

    data = datasets.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=0.5)

    ng = OKMEANS(num = 100, end = 100000)
    ng.train(data[0])

    plt.scatter(data[0][:,0], data[0][:,1])

    plt.scatter(ng.units[:,0], ng.units[:,1], s=50,c=(0.5,1,1))

    plt.savefig("kmeans.png")
