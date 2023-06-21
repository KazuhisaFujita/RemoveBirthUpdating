#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt
import networkx as nx
import sys
from sklearn import datasets
import matplotlib.pyplot as plt

class OKRB():
    def __init__(self, num = 100, dim = 2, end = 100000, lam = 0.1, rb_metric = "num_wins", th_rb = 0.3, beta = 0.0005):

        #dimension
        self.Dim = dim
        # learning rate
        self.lam = 0.1
        # max of units
        self.NUM = num
        # the number of delete processes
        self.END = end

        # Metric for RB updating
        self.Metric = rb_metric
        # threshold for RB processc
        self.Th_RB = th_rb
        # decay of error
        self.Beta = beta

        #RB updating counter
        self.counter_rb = 0

    def initialize_units(self):
        self.units = np.random.rand(self.NUM, self.Dim)
        self.wins = np.zeros(self.NUM)
        self.E = np.zeros(self.NUM)
        self.U = np.zeros(self.NUM)

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

    def RemoveCreateUnit(self, min_u, max_u):
        self.counter_rb += 1

        # Find the nearest neighbor units of the winning unit.
        f = np.linalg.norm(self.units - self.units[max_u], axis = 1).argsort()[1]
        # argsort()[0] is the winning unit itself.

        # Replace the variables of the unit with the minimum value.
        self.units[min_u] = (self.units[max_u] + self.units[f])/2
        self.wins[min_u]  = (self.wins[max_u]  + self.wins[f])/2
        self.E[min_u]     = (self.E[max_u]     + self.E[f])/2
        self.U[min_u]     = (self.U[max_u]     + self.U[f])/2

    def RBupdating(self, dists_x):
        # Remove and birth process

        # Find the winning unit and the second winning unit.
        n_win, n_win2  = dists_x.argsort()[[0, 1]]

        if self.Metric == "num_wins":
            # Use the number of wins for RB updating

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
            # Use the error for RB updating

            self.E[n_win] += dists_x[n_win]**2
            self.U[n_win] += dists_x[n_win2]**2 - dists_x[n_win]**2

            min_u = np.argmin(self.U)
            max_u = np.argmax(self.E)
            if self.U[min_u] / self.E[max_u] < self.Th_RB:
                self.RemoveCreateUnit(min_u, max_u)

            # Decrease all errors and utility measure.
            self.E -= self.Beta * self.E
            self.U -= self.Beta * self.U

    def learn(self, x, t):
        # Calculate distances between x and reference vectors.
        dists_x = np.linalg.norm(self.units - x, axis = 1)

        # Find a winning unit.
        n_win = dists_x.argmin()

        # Adaptation
        self.units[n_win] +=  self.lam * (x - self.units[n_win])

        #RB updating
        self.RBupdating(dists_x)


    def train(self, data):
        self.initialize_units()

        for t in range(1, self.END + 1):
            x = data[np.random.choice(data.shape[0])]
            self.learn(x, t)

if __name__ == '__main__':

    data = datasets.make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=0.5)

    ng = OKRB(num = 100, end = 100000, rb_metric = "error", th_rb = 0.3)
    ng.train(data[0])

    plt.scatter(data[0][:,0], data[0][:,1])

    plt.scatter(ng.units[:,0], ng.units[:,1], s=50,c=(0.5,1,1))

    plt.savefig("kmeans.png")
