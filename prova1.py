from abc import ABC
import math
import numpy as np
from sklearn.decomposition import pca
from sklearn.linear_model import logistic
import itertools as ittls
import collections


class Node:
    splitPca = "splitPca"
    splitLR = "splitLR"
    splitStd = "splitStd"

    def __init__(self, X, y, depth, splitMethod):
        # self.X = np.array(X)
        # self.y = np.array(y)
        self.X = X
        self.y = y # class should be 0, 1, 2, 3, ...
        self.depth = depth
        self.splitMethod = splitMethod
        self.decisionFun = None
        self.sons = []
        # frequency of each class in that node
        self.dictClasses = collections.Counter(self.y)
        # most frequent class
        self.clss = max(self.dictClasses, key=lambda x: self.dictClasses[x])
        self.nClasses = len(self.clss)
        self.nInst = len(self.X)
        self.nAttr = len(self.X[0])

    def predict(self, x):
        if self._isNodeLeaf():
            return self.clss
        else:
            son = self.decisionFun(x)
            return son.predict(x)

    def split(self):
        if self._stopCriteria():
            return 

        self.splitMethod.split()
        # if self.splitMethod == Node.splitPca:
        #     self.decisionFun = self._splitPca()
        # elif self.splitMethod == Node.splitLR:
        #     self.decisionFun = self._splitLr()
        # elif self.splitMethod == Node.splitStd:
        #     self.decisionFun = self.SplitStd
        # else:
        #     assert False
            
        self._physicalSplit()
        
        for son in self.sons:
            son.split()
            
    def _stopCriteria(self):
        if self.nInst < 20:
            return True
        elif self._gini([list(self.dictClasses.items())]) < 0.05:
            return True
        else:
            return False
            
    def _physicalSplit(self):
        dataSons = [{'X': [], 'y': []} for _ in range(2)]
        for (i, instance) in enumerate(self.X):
            idSon = self.decisionFun(instance)
            dataSons[idSon]['X'].append(instance)
            dataSons[idSon]['y'].append(self.y[i])
        
        self.sons = [Node(dataSons[i]['X'], dataSons[i]['y'], self.depth + 1, self.splitMethod) for i in range(2)]

    @staticmethod
    def _gini(matrix):
        totalCount = 0
        partialGini = 0
        for distr in matrix:
            s = sum(distr) + 0.0000001 # to avoid division by 0
            totalCount += s
            partialGini += sum((elem) * (1 - elem / s) for elem in distr)
        return partialGini / totalCount

    def _isNodeLeaf(self):
        return self.decisionFun is None


    # Inner classes for the different splits #

    class BaseSplit(ABC):
        def split(self, node):
            raise NotImplementedError("Should have implemented this")

    def _splitPca(self):
        pass

    def _splitLr(self):
        pass
    
    class SplitStd(BaseSplit):
        def split(self, node):
            bestGini = math.inf
            bestDecisionFun = None
            for idxAttr in range(node.nAttr):
                # TODO call a different function according the type of the attribute (numeric or nominal)
                gini, decisionFun = self._numericSplit(node, idxAttr)
                if gini < bestGini:
                    bestGini = gini
                    bestDecisionFun = decisionFun
            return bestDecisionFun


        def _numericSplit(self, node, idxAttr):
            sortedAttr = sorted(((node.X[i][idxAttr], node.y[i]) for i in range(node.nInst)))
            distrPerBranch = list(node.dictClasses.items())

            gini = node._gini(distrPerBranch)
            clss_ant = None
            attr_ant = None
            for (i, (attr, clss)) in enumerate(sortedAttr):
                distrPerBranch[0][clss] -= 1
                distrPerBranch[1][clss] += 1
                if clss_ant != clss and attr_ant != attr:
                    newGini = node._gini(distrPerBranch)
                    if newGini <= gini:
                        gini = newGini
                        splitPoint = attr
                clss_ant = clss
                attr_ant = attr

            return gini, lambda instance: instance[idxAttr] < splitPoint
