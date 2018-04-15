from functools import partial
import random as rnd
import inspect
from abc import ABC
import math
import numpy as np
from sklearn.decomposition import pca
from sklearn.linear_model import LogisticRegression
import itertools as ittls
import collections
import logging as log
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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


def calcEntropy(matrix):
    totalCount = 0
    partialEntropy = 0
    for distr in matrix:
        s = sum(distr) + 0.0000001  # to avoid division by 0
        totalCount += s
        assert all((elem >= 0 for elem in distr))
        assert s > 0
        partialEntropy -= sum(elem * math.log2((elem+0.0000001)/s) for elem in distr)
    return partialEntropy / totalCount



def calcCountsClasses(y, nClasses):
    listCounts = [0] * nClasses
    for clss in y:
        listCounts[clss] += 1
    return listCounts
    # dictClasses = {}
    # for clss in y:
    #     if clss in dictClasses:
    #         dictClasses[clss] += 1
    #     else:
    #         dictClasses[clss] = 1
    #
    # listCounts = [0] * (max(dictClasses) + 1)
    # for clss in range(len(listCounts)):
    #     if clss in dictClasses:
    #         listCounts[clss] = dictClasses[clss]
    # return listCounts


# The classes #

class DecisionTree:
    # string values of the class to distiguish the different methods available to split a node
    splitPca = "splitPca"
    splitLR = "splitLR"
    splitStd = "splitStd"

    # set log level
    def __init__(self, splitMethodName=splitStd, minGiniReduction=0.01, maxDepth=15, minNodeSize=20, minGini=0.05, splitCriteria=calcEntropy):
        # rnd.seed = 1
        self.node = None
        self.spliMethodName = splitMethodName

        if splitMethodName == DecisionTree.splitStd:
            self.splitMethod = Node.SplitStd()
        elif splitMethodName == DecisionTree.splitLR:
            self.splitMethod = Node.SplitLr()
        elif splitMethodName == DecisionTree.splitPca:
            self.splitMethod = Node.SplitPca()
        else:
            raise ValueError("Invalid splitMethod: " + splitMethodName)

        if minGiniReduction <= 0:
            raise ValueError("minGiniReduction must be greater than 0")
        self.minGiniReduction = minGiniReduction

        self.maxDepth = maxDepth
        self.minNodeSize = minNodeSize
        self.minGini = minGini
        self.splitCriteria = splitCriteria
        self.classToIdx = None
        self.idxToClass = None
        self.nClasses = None

    def __str__(self):
        if self.node is None:
            return "None"
        else:
            return self.node.__str__()

    def fit(self, X, y):
        y = self._transClassToIdx(y)
        self.nClasses = len(set(y))
        self.node = Node(X, y, self)
        self.node.split()

    def predict(self, X):
        assert self.node is not None
        yPredIdx = [self.node.predict(x) for x in X]
        yPredClss = self._transIdxToClass(yPredIdx)
        return yPredClss

    def predict_proba(self, X):
        assert self.node is not None
        return [self.node.predict_proba(x) for x in X]

    def score(self, X, y):
        yPred = self.predict(X)
        return accuracy_score(y, yPred)

    def get_params(self, deep=True):
        # "node": self.node,
        return {"splitMethodName": self.spliMethodName, "minGiniReduction": self.minGiniReduction,
                "maxDepth": self.maxDepth, "minNodeSize": self.minNodeSize, "minGini": self.minGini}

    def _transClassToIdx(self, y):
        """
        Crates two dictionaries, one to transform classes to indices and the other to transform indices to classes.
        Then, it return the y, a list of classes of any type to a list of numbers, being each number 0 <= n < nClasses
        :param y: A list where each element represents a class
        :return: A list where each element is an index (number) representing a class
        """
        tupleIdxToClass = list(enumerate(set(y)))
        tupleClassToIdx = [(b,a) for (a,b) in tupleIdxToClass]
        self.classToIdx = dict(tupleClassToIdx)
        self.idxToClass = dict(tupleIdxToClass)

        return [self.classToIdx[clss] for clss in y]

    def _transIdxToClass(self, y):
        assert self.idxToClass is not None
        return [self.idxToClass[idx] for idx in y]


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
        self.freqClasses = calcCountsClasses(self.y, self.parentTree.nClasses)
        assert all(freq >= 0 for freq in self.freqClasses)
        # most frequent class
        self.clss = self.freqClasses.index(max(self.freqClasses))
        self.nClassesNode = sum(freq != 0 for freq in self.freqClasses)
        self.nInst = len(self.X)
        self.nAttr = len(self.X[0])
        self.gini = self.parentTree.splitCriteria([self.freqClasses])
        self.accuracy = max(self.freqClasses)/sum(self.freqClasses)

    def __str__(self):
        # La accuracy s'ha de generalitza per a datasets amb etiquetes diferents a True i False
        # "{0:.2f}".format(a)
        giniSons = sum((self.parentTree.splitCriteria([son.freqClasses])*len(son.X) for son in self.sons)) / len(self.X)
        strTree = 'size: ' + str(self.nInst) + '; Accuracy: ' + \
                  '{0:.2f}'.format(self.accuracy) + \
                  '; GiniFather: ' + '{0:.2f}'.format(self.parentTree.splitCriteria([self.freqClasses])) + \
                  "; GiniSons: " + '{0:.2f}'.format(giniSons) + "; Predict: " + str(self.clss) + '\n'
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
            idSon = self.decisionFun.apply(x)
            return self.sons[idSon].predict(x)

    def predict_proba(self, x):
        """
        Predicts the class for the instance x
        :param x: x is a single list or np.array of 1xn
        :return: The predicted value for x
        """
        if self._isNodeLeaf():
            # it should know the real number of classes, because in a node may not appear instances of a certain class
            return [freq/len(self.y) for freq in self.freqClasses]
        else:
            idSon = self.decisionFun.apply(x)
            return self.sons[idSon].predict_proba(x)

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
            self.decisionFun = None
            return

        # After setting the decision function, it performs the split physically. That mean it generated the new sons
        self._physicalSplit(nSubsets)

        # Apply recursively the split() to each new son
        for son in self.sons:
            son.split()

    def _stopCriteria(self):
        """
        Function that returns True if the node should be a leaf and, therefore, should not be splitted
        :return: True or False
        """
        if self.nInst < self.parentTree.minNodeSize:
            return True
        elif self.gini < self.parentTree.minGini:
            return True
        elif self.depth > self.parentTree.maxDepth:
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
            idSon = self.decisionFun.apply(instance)
            dataSons[idSon]['X'].append(instance)
            dataSons[idSon]['y'].append(self.y[i])

        assert all(len(data['X']) > 0 for data in dataSons)
        # Create each new son if there's data for it
        self.sons = [Node(dataSons[i]['X'], dataSons[i]['y'], self.parentTree, self.depth + 1)
                     for i in range(nSubsets) if len(dataSons[i]['X']) > 0]
        assert len(self.sons) >= 2


    def _isNodeLeaf(self):
        return self.decisionFun is None


    # Inner classes for the different splits #

    class BaseSplit(ABC):
        def split(self, node):
            raise NotImplementedError("Should have implemented this")

        @classmethod
        def _numericSplit(cls, X, y, idxAttr, node):
            sortedAttr = sorted(((X[i][idxAttr], y[i]) for i in range(node.nInst)))
            distrPerBranch = [node.freqClasses.copy(), [0]*node.nClassesNode]

            bestGini = math.inf
            (attr_ant, clss_ant) = (sortedAttr[0][0], sortedAttr[0][1])
            splitPoint = None
            for (attr, clss) in sortedAttr[1:]:
                distrPerBranch[0][clss_ant] -= 1
                distrPerBranch[1][clss_ant] += 1
                if attr_ant != attr:  # and class_ant != clss
                    newGini = node.parentTree.splitCriteria(distrPerBranch)
                    bestGini, splitPoint = min((newGini, (attr_ant+attr)/2), (bestGini, splitPoint))
                (attr_ant, clss_ant) = (attr, clss)

            # TODO Instead of returning a lambda, I could return an object of class DecisionFunction that can bring more information about the split
            return bestGini, cls.BaseSplitDecisionFun(idxAttr, splitPoint)  # lambda instance: instance[idxAttr] <= splitPoint

        class BaseSplitDecisionFun:
            def __init__(self, idxAttr, splitPoint):
                self.idxAttr = idxAttr
                self.splitPoint = splitPoint

            def apply(self, instance):
                return instance[self.idxAttr] <= self.splitPoint

    class SplitPca(BaseSplit):
        def __init__(self):
            # TODO Take the necessary components until a minimum variance is explained or until no improvement in gini is found
            self.nComp = 4
            self.pca = pca.PCA(n_components=self.nComp)

        def split(self, node):
            # TODO What happens when there are categorical attributes?
            scaler = StandardScaler().fit(node.X)
            xScaled = scaler.transform(node.X)
            self.pca.fit(xScaled)
            xProj = self.pca.transform(xScaled)

            bestGini, bestDecisionFun = math.inf, None
            for idxComp in range(self.nComp):
                gini, decisionFun = self._numericSplit(xProj, node.y, idxComp, node)
                bestGini, bestDecisionFun = min((gini, decisionFun), (bestGini, bestDecisionFun), key=lambda x: x[0])

            copyPca = copy.copy(self.pca)
            return bestGini, 2, self.SplitPcaDecisionFun(copyPca, scaler, bestDecisionFun)

        class SplitPcaDecisionFun:
            def __init__(self, pca, scaler, decFun):
                self.pca = pca
                self.scaler = scaler
                self.decFun = decFun

            def apply(self, instance):
                return self.decFun.apply(self.pca.transform(self.scaler.transform([instance]))[0])

    class SplitLr(BaseSplit):
        def __init__(self):
            self.nVars = 2
            self.lr = LogisticRegression(C=999999)

        def split(self, node):
            nIter = node.nAttr
            (bestGini, bestDecisionFun, bestDistr) = (math.inf, None, None)
            for i in range(node.nAttr):
                for j in range(i+1, node.nAttr):
                    # select some attributes randomly
                    # idxAttr = [rnd.randint(0, node.nAttr-1) for _ in range(self.nVars)]
                    idxAttr = [i, j]
                    xAux = [[instance[attr] for attr in idxAttr] for instance in node.X]
                    self.lr.fit(xAux, node.y)
                    yPred = self.lr.predict(xAux)
                    # each predicted class represents a future branch
                    # we need to know, for each instance, in which class it actually belongs to
                    distrPerBranch = [[0]*node.nClassesNode for _ in range(node.nClassesNode)]
                    for k in range(len(yPred)):
                        distrPerBranch[yPred[k]][node.y[k]] += 1

                    # TODO It should support an split that gives some empty branches, but not only one branch
                    if all((sum(branch) > 0 for branch in distrPerBranch)):
                        gini = node.parentTree.splitCriteria(distrPerBranch)
                        copyLr = copy.copy(self.lr)
                        # decisionFun = lambda x: copyLr.predict(self._takeAttr([x], idxAttr))[0]
                        decisionFun = self.SplitLrDecisionFun(copyLr, idxAttr)
                        (bestGini, bestDecisionFun, bestDistr) = min((gini, decisionFun, distrPerBranch),
                                                                     (bestGini, bestDecisionFun, bestDistr), key=lambda x: x[0])

            return bestGini, node.nClassesNode, bestDecisionFun

        class SplitLrDecisionFun:
            def __init__(self, lr, listAttr):
                self.lr = lr
                self.listAttr = listAttr

            def apply(self, instance):
                attr = [instance[attr] for attr in self.listAttr]
                return self.lr.predict([attr])[0]


    class SplitStd(BaseSplit):
        # TODO It's better to maintain sorted all the attributes
        # TODO One vs all split?
        def split(self, node):
            bestGini, bestDecisionFun = math.inf, None
            for idxAttr in range(node.nAttr):
                # TODO call a different function according the type of the attribute (numeric or nominal)
                gini, decisionFun = self._numericSplit(node.X, node.y, idxAttr, node)
                bestGini, bestDecisionFun = min((gini, decisionFun), (bestGini, bestDecisionFun), key=lambda x: x[0])
            # TODO parametrize the number of subsets it returns
            return bestGini, 2, bestDecisionFun
