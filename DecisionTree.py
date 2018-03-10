import inspect
from abc import ABC
import math
import numpy as np
from sklearn.decomposition import pca
from sklearn.linear_model import logistic
import itertools as ittls
import collections
import logging as log

# Auxiliar functions #

def calcGini(matrix):
    """
    Calculates the gini cost of the matrix. For example, each row contains the data that will eventually
    belong to a new node. Each position of the row represents a class and its corresponding number
    represents the frequency of that class.
    [[5, 10, 7] -> For son0, class0 appears 5 times, class1 10 times and class2 7 times
    [8, 15, 2]] -> For son1, class0 appears 8 times, class1 15 times and class2 2 times
    :param matrix: Matrix of the frequencies of each class for each future son
    :return: The gini cost
    """
    totalCount = 0
    partialGini = 0
    for distr in matrix:
        s = sum(distr) + 0.0000001  # to avoid division by 0
        totalCount += s
        partialGini += sum(elem * (1 - elem / s) for elem in distr)
    return partialGini / totalCount

# The classes #

class DecisionTree:
    # string values of the class to distiguish the different methods available to split a node
    splitPca = "splitPca"
    splitLR = "splitLR"
    splitStd = "splitStd"

    # set log level
    def __init__(self, splitMethod=splitStd, minGiniReduction=0.01):
        self.node = None

        if splitMethod == DecisionTree.splitStd:
            self.splitMethod = Node.SplitStd()
        elif splitMethod == DecisionTree.splitLR:
            self.splitMethod = Node.SplitLr()
        elif splitMethod == DecisionTree.splitPca:
            self.splitMethod = Node.SplitPca()
        else:
            raise ValueError("Invalid splitMethod: " + splitMethod)

        if minGiniReduction <= 0:
            raise ValueError("minGiniReduction must be greater than 0")
        self.minGiniReduction = minGiniReduction

    def __str__(self):
        if self.node is None:
            return "None"
        else:
            return self.node.__str__()

    def fit(self, X, y):
        self.node = Node(X, y, self)
        self.node.split()

    def predict(self, X):
        assert self.node is not None
        return [self.node.predict(x) for x in X]

# Should Node be an inner class of DecisionTree?

class Node:
    def __init__(self, X, y, parentTree, depth=0):
        """
        :param X: Matrix with the data. Each row represents an individual (instance)
        :param y: Class values
        :param depth: Depth of the node in the overall tree
        :param splitMethod: Method used to split a node
        """
        self.X = X
        self.y = y # class should be 0, 1, 2, 3, ...
        self.depth = depth
        self.parentTree = parentTree
        self.decisionFun = None
        self.sons = []
        # frequency of each class in that node
        self.freqClasses = self._calcFreqClasses()
        # most frequent class
        self.clss = self.freqClasses.index(max(self.freqClasses))
        self.nClasses = len(self.freqClasses)
        self.nInst = len(self.X)
        self.nAttr = len(self.X[0])
        self.gini = calcGini([self.freqClasses])
        self.accuracy = max(self.freqClasses)/sum(self.freqClasses)

    def __str__(self):
        # La accuracy s'ha de generalitza per a datasets amb etiquetes diferents a True i False
        # "{0:.2f}".format(a)
        strTree = 'size: ' + str(self.nInst) + '; Accuracy: ' + \
                  '{0:.2f}'.format(self.accuracy) + \
                  '; Gini: ' + '{0:.2f}'.format(calcGini([self.freqClasses])) + "; Predict: " + str(self.clss) + '\n'
        # posar les funcions de accuracy i altres (recall, precision...) fora de la classe i
        # cridar-les en aquest print
        for i in range(len(self.sons)):
            strTree += (self.depth + 1) * '\t' + str(i) + ' -> ' + self.sons[i].__str__()
        return strTree

    def predict(self, x):
        """
        Predicts the class for the instance x
        :param x: x is a single list or np.array of 1xn
        :return: The predicted value for x
        """
        if self._isNodeLeaf():
            return self.clss
        else:
            idSon = self.decisionFun(x)
            return self.sons[idSon].predict(x)

    def split(self):
        """
        Splits this node and creates two sons if the node can be splited. Otherwise, it does nothing
        """
        if self._stopCriteria():
            return

        # Calculate the decision function using the split method of this node
        # The decision function is a function that given an instance it return in which son this instance should go
        giniSons, nSubsets, self.decisionFun = self.parentTree.splitMethod.split(self)

        # Avoid spliting if we don't reduce the gini score
        if giniSons + self.parentTree.minGiniReduction >= self.gini:
            return

        # After setting the decision function, it performs the split physically. That mean it generated the new sons
        self._physicalSplit(nSubsets)

        # Apply recursively the split() to each new son
        for son in self.sons:
            son.split()

    def _calcFreqClasses(self):
        dictClasses = {}
        for clss in self.y:
            if clss in dictClasses:
                dictClasses[clss] += 1
            else:
                dictClasses[clss] = 1

        listFreq = [0] * (max(dictClasses) + 1)
        for clss in range(len(listFreq)):
            if clss in dictClasses:
                listFreq[clss] = dictClasses[clss]
        return listFreq

    def _stopCriteria(self):
        """
        Function that returns True if the node should be a leaf and, therefore, should not be splitted
        :return: True or False
        """
        if self.nInst < 20:
            return True
        elif self.gini < 0.05:
            return True
        elif self.depth > 15:
            return True
        else:
            return False
            
    def _physicalSplit(self, nSubsets):
        """
        Performs the split of the data using its decision function (self.decisionFun) and creates the new sons
        """
        assert self.decisionFun is not None
        # In dataSons[i] we will have the X and y values for the future "i" son
        dataSons = [{'X': [], 'y': []} for _ in range(nSubsets)]
        for (i, instance) in enumerate(self.X):
            idSon = self.decisionFun(instance)
            dataSons[idSon]['X'].append(instance)
            dataSons[idSon]['y'].append(self.y[i])

        # assert all(len(data['X']) > 0 for data in dataSons)
        # Create each new son if there's data for it
        self.sons = [Node(dataSons[i]['X'], dataSons[i]['y'], self.parentTree, self.depth + 1)
                     for i in range(nSubsets) if len(dataSons[i]['X']) > 0]


    def _isNodeLeaf(self):
        return self.decisionFun is None


    # Inner classes for the different splits #

    class BaseSplit(ABC):
        def split(self, node):
            raise NotImplementedError("Should have implemented this")

    class SplitPca(BaseSplit):
        def split(self, node):
            raise NotImplementedError("Should have implemented this")

    class SplitLr(BaseSplit):
        def split(self, node):
            raise NotImplementedError("Should have implemented this")
    
    class SplitStd(BaseSplit):
        def split(self, node):
            bestGini = math.inf
            bestDecisionFun = None
            for idxAttr in range(node.nAttr):
                # TODO call a different function according the type of the attribute (numeric or nominal)
                gini, decisionFun = self._numericSplit(node, idxAttr)
                bestGini, bestDecisionFun = min((gini, decisionFun), (bestGini, bestDecisionFun), key=lambda x: x[0])
            # TODO parametrize the number of subsets it returns
            return bestGini, 2, bestDecisionFun

        @staticmethod
        def _numericSplit(node, idxAttr):
            sortedAttr = sorted(((node.X[i][idxAttr], node.y[i]) for i in range(node.nInst)))
            distrPerBranch = [node.freqClasses.copy(), [0]*node.nClasses]

            bestGini = math.inf
            (attr_ant, clss_ant) = (sortedAttr[0][0], sortedAttr[0][1])
            splitPoint = None
            for (attr, clss) in sortedAttr[1:]:
                distrPerBranch[0][clss_ant] -= 1
                distrPerBranch[1][clss_ant] += 1
                if attr_ant != attr:  # and class_ant != clss
                    newGini = calcGini(distrPerBranch)
                    bestGini, splitPoint = min((newGini, (attr_ant+attr)/2), (bestGini, splitPoint))
                (attr_ant, clss_ant) = (attr, clss)

            # TODO Instead of returning a lambda, I could return an object of class DecisionFunction that can bring more information about the split
            return bestGini, lambda instance: instance[idxAttr] <= splitPoint
