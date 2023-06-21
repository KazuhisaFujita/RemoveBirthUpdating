#---------------------------------------
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
import math as mt
import pylab as plt
import networkx as nx
from sklearn import cluster, datasets

class SOMRB:
    def __init__(self, num = 100, dim = 2, end = 100000, rate = 0.1, sigma = 1, sig_kernel = 0.5, rb_metric = "num_wins", th_rb = 0.01, beta = 0.0005):

        # dimension
        self.Dim = dim
        # the number of untis
        self.NUM = num
        # the side of 2dimensional array.
        self.side = mt.floor(mt.sqrt(num))
        # end step
        self.END = end
        # learning rate
        self.rate = rate
        # a parameter of neighborhood function
        self.sigma = sigma

        # Metric for RB updating
        self.Metric = rb_metric
        # threshold for RB processc
        self.Th_RB = th_rb
        # decay of error
        self.Beta = beta

        self.sig_kernel = sig_kernel

        #RB updating counter
        self.counter_rb = 0

    def initialize_units(self):
        # initialize the positions of the unit on feature space.

        # Calculate the initial weights.
        self.units = np.zeros((self.NUM, self.Dim))
        for i in range(self.NUM):
            self.units[i][0] = (i//self.side)/self.side
            self.units[i][1] = (i%self.side)/self.side

        # Create a graph.
        self.g_units = nx.Graph()
        for i in range(self.NUM):
            self.g_units.add_node(i)

        # Calculate the initial position of a unit on the 2dimensional space.
        self.pos = np.zeros((self.NUM, 2), dtype=np.int64)
        for i in range(self.NUM):
            self.pos[i][0] = i//self.side
            self.pos[i][1] = i%self.side

        # Connect neighbor units.
        for i in range(self.NUM):
            for j in range(i + 1, self.NUM):
                if   self.pos[j][0] == self.pos[i][0] - 1 and self.pos[j][1] == self.pos[i][1]:
                    self.g_units.add_edge(j, i, weight = 0)
                elif self.pos[j][0] == self.pos[i][0]     and self.pos[j][1] == self.pos[i][1] - 1:
                    self.g_units.add_edge(j, i, weight = 0)
                elif self.pos[j][0] == self.pos[i][0] + 1 and self.pos[j][1] == self.pos[i][1]:
                    self.g_units.add_edge(j, i, weight = 0)
                elif self.pos[j][0] == self.pos[i][0]     and self.pos[j][1] == self.pos[i][1] + 1:
                    self.g_units.add_edge(j, i, weight = 0)

        # The number of wins
        self.wins = np.zeros(self.NUM)

        # Error variables
        self.E = np.zeros(self.NUM)
        self.U = np.zeros(self.NUM)


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
        count = np.zeros(self.NUM)
        for i in data:
            count[np.argmin(self.dists(i, self.units))] += 1
        return(np.count_nonzero(count == 0))

    def neig_func(self, dist):
        return np.exp(- (dist**2) / 2 / (self.sigma**2))

    def WeightAdaptation(self, x):
        dists_x = np.linalg.norm(self.units - x, axis = 1)

        # Find the winning unit.
        min_unit_num = dists_x.argmin()

        # Calculate the distances between units and the winning unit on the 2D space.
        dists_p = np.linalg.norm(self.pos - self.pos[min_unit_num], axis=1)
        self.units += self.rate * np.multiply((x - self.units), self.neig_func(dists_p).reshape(self.NUM,1))

    def AddEdges(self, min_u):
        for i in range(self.NUM):
            if   self.pos[i][0] == self.pos[min_u][0] - 1 and self.pos[i][1] == self.pos[min_u][1]:
                self.g_units.add_edge(min_u, i, weight = 0)
            elif self.pos[i][0] == self.pos[min_u][0]     and self.pos[i][1] == self.pos[min_u][1] - 1:
                self.g_units.add_edge(min_u, i, weight = 0)
            elif self.pos[i][0] == self.pos[min_u][0] + 1 and self.pos[i][1] == self.pos[min_u][1]:
                self.g_units.add_edge(min_u, i, weight = 0)
            elif self.pos[i][0] == self.pos[min_u][0]     and self.pos[i][1] == self.pos[min_u][1] + 1:
                self.g_units.add_edge(min_u, i, weight = 0)

    def CalWeight(self, min_u):
        # List the neighbors of the unit with the minimum wins.
        min_neighbors = list(self.g_units.neighbors(min_u))

        if len(min_neighbors) > 1:
            # The reference vector of the unit with the minimum wins is mean of the reference vectors of its neighbor units.
            self.units[min_u] = np.mean(self.units[min_neighbors], axis=0)
            self.wins[min_u]  = np.mean(self.wins[min_neighbors])
            self.E[min_u]     = np.mean(self.E[min_neighbors])
            self.U[min_u]     = np.mean(self.U[min_neighbors])

        else:# If the number of neighbor units of the unit with the minimum wins is 1
            neighbor = min_neighbors[0]
            # List the neighbors of the neighbor unit.
            neig = list(self.g_units.neighbors(neighbor))
            # Remove the unit with minimum wins from the list.
            neig.remove(min_u)

            if len(neig) != 0:
                neig.append(neighbor)
                self.units[min_u] = np.mean(self.units[neig], axis=0)
                self.wins[min_u]  = np.mean(self.wins[neig])
                self.E[min_u]     = np.mean(self.E[neig])
                self.U[min_u]     = np.mean(self.U[neig])
            else:# If the neighbor of the neighbor is only the units with minimum wins.
                # Find the nearest neighbor units of the neighbor
                f = np.linalg.norm(self.units - self.units[neighbor], axis = 1).argsort()[1]

                # Replace the variables of the unit with the minimum value.
                self.units[min_u] = (self.units[neighbor] + self.units[f])/2
                self.wins[min_u]  = (self.wins[neighbor]  + self.wins[f])/2
                self.E[min_u]     = (self.E[neighbor]     + self.E[f])/2
                self.U[min_u]     = (self.U[neighbor]     + self.U[f])/2


    def AddNode(self, min_u, max_u):
        # Add a new unit.
        self.g_units.add_node(min_u)

        # Find an empty vertex neighboring the unit with the most wins on the 2D grid.
        max_neighbors = list(self.g_units.neighbors(max_u))

        direction = [[0,1], [1,0], [0, -1], [-1,0]]

        list_empty = []
        for i in range(4):
            posx = self.pos[max_u][0] + direction[i][0]
            posy = self.pos[max_u][1] + direction[i][1]

            if not(any(np.all(row == np.array([posx,posy])) for row in self.pos)):
                list_empty.append([posx, posy])

        # Randomly select one from the empty vertexs.
        flag = np.random.randint(len(list_empty))

        # Set the position of the new unit.
        self.pos[min_u][0] = list_empty[flag][0]
        self.pos[min_u][1] = list_empty[flag][1]


    def RemoveCreateUnit(self, min_u, max_u):
        self.counter_rb += 1

        # Remove the unit with the minimum wins.
        self.g_units.remove_node(min_u)
        # Add a new unit.
        self.AddNode(min_u, max_u)
        # Add the edge between the new unit and its neighborhoods.
        self.AddEdges(min_u)
        # Calculate the variables of the new units.
        self.CalWeight(min_u)

    def RemoveBirth(self, x):
        # Remove and birth process

        list_degree = self.g_units.degree
        dists_x = self.dists(x, self.units)


        if self.Metric == "num_wins":
            n_win = dists_x.argmin()
            # Increment the number of wins by one
            self.wins[n_win] += 1

            # Find the units with the most wins and the minimum wins
            max_u = None
            max_win = float('-inf')

            for i in range(self.NUM):
                # Find the winning unit
                n_win = dists_x.argmin()

                if (list_degree[i] < 4) and (self.wins[i] > max_win):
                    max_win = self.wins[i]
                    max_u = i

            min_u = np.argmin(self.wins)

            if self.wins[max_u] != 0: # Avoid division by zero when the numbers of wins of all the unit on the edge of network are zero.
                if self.wins[min_u]/self.wins[max_u] < self.Th_RB:
                    self.RemoveCreateUnit(min_u, max_u)

            # Decay
            self.wins -= self.Beta * self.wins

        elif self.Metric == "error":
            # Find the winning unit and the second winning unit.
            n_win, n_win2 = dists_x.argsort()[[0,1]]

            self.E[n_win] += dists_x[n_win]**2
            self.U[n_win] += dists_x[n_win2]**2 - dists_x[n_win]**2

            max_u = None
            Emax = float('-inf')
            min_u = None
            Umin = float('inf')

            for i in range(self.NUM):
                if (list_degree[i] < 4) and (self.E[i] > Emax):
                    Emax = self.E[i]
                    max_u = i

                if self.U[i] < Umin:
                    Umin = self.U[i]
                    min_u = i

            if self.E[max_u] != 0 and max_u != min_u:
                if self.U[min_u] / self.E[max_u] < self.Th_RB:
                    self.RemoveCreateUnit(min_u, max_u)

            # Decrease all errors and utility measure.
            self.E -= self.Beta * self.E
            self.U -= self.Beta * self.U

    def learn(self, x, t):
        self.WeightAdaptation(x)
        self.RemoveBirth(x)

    def train(self, data):
        self.initialize_units()
        for t in range(1, self.END + 1):
            x = data[np.random.choice(data.shape[0])]
            self.learn(x, t)

if __name__ == '__main__':
    n_samples = 1500
    data = datasets.make_blobs(n_samples = n_samples)[0]
 
    kohonen = SOMRB(end = 100000, rb_metric="num_wins", rate = 0.2, sigma = 0.5, th_rb = 0.01, beta = 0.005)

    kohonen.train(data)
    plt.scatter(data[:,0], data[:,1])

    print(kohonen.TotalError(data))
    nx.draw_networkx_nodes(kohonen.g_units,kohonen.units,node_size=5,node_color=(0.5,1,1))
    nx.draw_networkx_edges(kohonen.g_units,kohonen.units,width=5,edge_color='b',alpha=0.5)
    plt.savefig("somrb.png")
