#---------------------------------------
#Since : 2018/07/05
#Update: 2023/05/26
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from sklearn import datasets
import torch
import torchvision
import torchvision.transforms as transforms

class Datasets(object):
    def __init__(self, n_samples = 1000, k = 3, dim = 2):
        # Set Parameters
        self.n_samples = n_samples
        self.k = k
        self.dim = dim

    def dataset(self, title):
        if title == "Square":
            k = 1
            n_samples = self.n_samples
            labels = np.zeros(n_samples)
            return k, np.random.rand(n_samples, 2), labels

        elif title == "Blobs":
            n_samples = self.n_samples
            k = self.k
            dim = self.dim
            data, labels = datasets.make_blobs(n_samples=n_samples, centers = k, n_features = dim)
            return k, data, labels

        elif title == "Circles":
            n_samples = self.n_samples
            k = 2
            data, labels = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
            return k, data, labels

        elif title == "Moons":
            n_samples = self.n_samples
            k = 2
            data, labels = datasets.make_moons(n_samples=n_samples, noise=.05)
            return k, data, labels

        elif title == "Swiss_roll":
            n_samples = self.n_samples
            k = 1
            labels = np.ones(n_samples)
            data, _ = datasets.make_swiss_roll(n_samples=n_samples, noise=.05)
            return k, data, labels

        elif title == "S_curve":
            n_samples = self.n_samples
            k = 1
            labels = np.ones(n_samples)
            data, _ = datasets.make_s_curve(n_samples=n_samples, noise=.05)
            return k, data, labels

        elif title == "Iris":
            k = 4
            data = datasets.load_iris()
            return k, data.data, data.target

        elif title == "Wine":
            k = 3
            data = datasets.load_wine()
            return k, data.data, data.target

        elif title == "spam":
            k = 2
            file = "datasets/spambase.data"
            data = np.loadtxt(file,delimiter=",")

            last = np.size(data[0]) - 1
            labels = data[:,last]

            X = np.delete(data, last, axis=1)
            X = np.delete(X,np.size(X[0])-1, axis=1)
            X = np.delete(X,np.size(X[0])-1, axis=1)
            X = np.delete(X,np.size(X[0])-1, axis=1)
            return k, X, labels

        elif title == "digits":
            digits = datasets.load_digits()
            X = digits.data
            k = 10
            labels = digits.target
            return k, np.array(X), np.array(labels)

        elif title == "MNIST_trainset":
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
            train_images = trainset.data.view(trainset.data.size(0),-1).numpy()
            train_labels = trainset.targets.numpy()
            k = 10
            return k, train_images, train_labels

        elif title == "MNIST_testset":
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
            test_images = testset.data.view(testset.data.size(0),-1).numpy()
            test_labels = testset.targets.numpy()
            k = 10
            return k, test_images, test_labels

        elif title == "olivetti":
            faces = datasets.fetch_olivetti_faces()
            X = faces.data
            k = 40
            labels = faces.target
            return k, np.array(X), np.array(labels)

        elif title == "CNAE":
            file = "datasets/CNAE-9.data"
            k = 9
            data = np.loadtxt(file,delimiter=",")
            last = np.size(data[0]) - 1
            label_true = data[:,0]
            X = np.delete(data, 0, axis=1)
            return k, X, label_true

        elif title == "ecoli":
            file = "datasets/ecoli.data"
            k = 8
            DATA = np.genfromtxt(file, dtype='str')

            X = np.zeros(np.shape(DATA))
            X = np.delete(X, 0, axis=1)
            X = np.delete(X, 0, axis=1)
            for i in range(1,np.size(DATA[0]) - 1):
                X[:, i - 1] = DATA[:, i].astype(np.float64)

            label_true = np.zeros(np.shape(X)[0])
            count = 0
            for i in DATA[:, np.size(DATA[0])-1]:
                if i == "cp":
                    label_true[count] = 0
                elif i == "im":
                    label_true[count] = 1
                elif i == "pp":
                    label_true[count] = 2
                elif i == "imU":
                    label_true[count] = 3
                elif i == "om":
                    label_true[count] = 4
                elif i == "omL":
                    label_true[count] = 5
                elif i == "imL":
                    label_true[count] = 6
                elif i == "imS":
                    label_true[count] = 7
                count+=1

            return k, X, label_true

        elif title == "glass":
            file = "datasets/glass.data"
            k = 7
            X = np.loadtxt(file,delimiter=",")
            X = np.delete(X, 0, axis=1)

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "yeast":
            file = "datasets/yeast.data"
            k = 10
            DATA = np.genfromtxt(file, dtype='str')

            X = np.zeros(np.shape(DATA))
            X = np.delete(X, 0, axis=1)
            X = np.delete(X, 0, axis=1)

            for i in range(1,np.size(DATA[0]) - 1):
                X[:, i - 1] = DATA[:, i].astype(np.float64)

            label_true = np.zeros(np.shape(X)[0])
            count = 0
            for i in DATA[:, np.size(DATA[0])-1]:
                if i == "CYT":
                    label_true[count] = 0
                if i == "NUC":
                    label_true[count] = 1
                elif i == "MIT":
                    label_true[count] = 2
                elif i == "ME3":
                    label_true[count] = 3
                elif i == "ME2":
                    label_true[count] = 4
                elif i == "ME1":
                    label_true[count] = 5
                elif i == "EXC":
                    label_true[count] = 6
                elif i == "VAC":
                    label_true[count] = 7
                elif i == "POX":
                    label_true[count] = 8
                elif i == "ERL":
                    label_true[count] = 9
                count+=1

            return k, X, label_true

        elif title == "balance":
            file = "datasets/balance-scale.data"
            k = 3
            DATA = np.genfromtxt(file,delimiter=",", dtype='str')

            X = np.zeros(np.shape(DATA))
            X = np.delete(X, 0, axis=1)

            for i in range(1,np.size(DATA[0]) - 1):
                X[:, i - 1] = DATA[:, i].astype(np.float64)

            label_true = np.zeros(np.shape(X)[0])
            count = 0
            for i in DATA[:, 0]:
                if i == "L":
                    label_true[count] = 0
                if i == "B":
                    label_true[count] = 1
                elif i == "R":
                    label_true[count] = 2
                count+=1

            return k, X, label_true

        elif title == "Aggregation":
            file = "datasets/Aggregation.txt"
            k = 7
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "Compound":
            file = "datasets/Compound.txt"
            k = 6
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "Pathbased":
            file = "datasets/pathbased.txt"
            k = 3
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "Spiral":
            file = "datasets/spiral.txt"
            k = 3
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "D31":
            file = "datasets/D31.txt"
            k = 31
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "R15":
            file = "datasets/R15.txt"
            k = 15
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "Jain":
            file = "datasets/jain.txt"
            k = 2
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "Flame":
            file = "datasets/flame.txt"
            k = 2
            X = np.loadtxt(file,delimiter="\t")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "t4.8k":
            file = "datasets/t4.8k.txt"
            k = 6
            X = np.loadtxt(file)
            label_true = []
            return k, X, label_true

        elif title == "Complex9":
            file = "datasets/complex9.txt"
            k = 9
            X = np.loadtxt(file, delimiter=",")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1)

            return k, X, label_true

        elif title == "t7.10k":
            file = "datasets/t7.10k.csv"
            k = 9

            X = np.loadtxt(file, delimiter=",", dtype = "unicode")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1).astype(np.float32)

            return k, X, label_true

        elif title == "t8.8k":
            file = "datasets/t8.8k.csv"
            k = 9

            X = np.loadtxt(file, delimiter=",", dtype = "unicode")

            label_true = X[:,np.size(X[0]) - 1]
            X = np.delete(X, np.size(X[0]) - 1, axis=1).astype(np.float32)

            return k, X, label_true


if __name__ == '__main__':
    k, X, labels = Datasets().dataset("MNIST_trainset")
    print(k)
    print(X.shape)
    print(labels)
    k, X, labels = Datasets().dataset("digits")
    print(k)
    print(X.shape)
    print(labels)
