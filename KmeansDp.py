import math
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score

class KmeansDp:
    def __init__(self, nClusters = 2):
        self.nClusters = nClusters
        self.kmeans = KMeans()
        self.X = None

    def fit(self, X, optimum=False):
        # TODO Check dimensionality of X
        self.X = X
        # There cannot be more clusters than different values in the dataset
        if self.nClusters > len(set(self.X)):
            raise ValueError("The number of clusters is greater than the number of different elements in X")

        self.X.sort()
        nElems = len(self.X)

        # listVar[i] contains the variance of the cluster formed by the elements in the range [0, i]
        listVar = self._initListVar()
        # We will find the optimum clustering for k=2 clusters to k = maxClusters - 1
        iniClust = [[0] for _ in range(nElems)]
        lastScore = -1

        for k in range(2, self.nClusters + 1):
            listVar, iniClust = self._computeVariance(k, listVar, iniClust)

            if optimum:
                # Evaluate the goodness of the clustering
                auxKmeans = KMeans()
                cl = auxKmeans.predict(self.X)
                score = silhouette_score(self.X, cl)
                if lastScore >= score:
                    # No increase in goodness of the clustering
                    break
                self.kmeans = auxKmeans

        if not optimum:
            # Calculate the cluster centers from the limit points
            self.kmeans.cluster_centers_ = self._calculateCenters(self.nClusters, iniClust)

    def predict(self, X):
        return self.kmeans.predict(X)
        
    def _initListVar(self):
        nElems = len(self.X)
        mean = self.X[0]
        listVar = [0] * nElems
        for i in range(1, nElems):
            # "i" points to the next element that we'll add to the clustering, but it's also the number
            # of elements we have in our clustering (listVar[i])
            newMean = (i * mean + self.X[i]) / (i + 1)
            # The new variance is the change of the variance of the current points in [0, i-1] due to the change
            # of the mean (newMean - mean)**2 plus the variance the new point "i" adds to the overall clustering
            listVar[i] = listVar[i - 1] + i * (newMean - mean) ** 2 + (self.X[i] - newMean) ** 2
            mean = newMean

    def _computeVariance(self, k, listVar, iniClust):
        nElems = len(self.X)
        listVarNew = [0] * nElems
        iniClustNew = [[0] for _ in range(nElems)]
        # For each sublist of X that goes from 0 to i, [0, i], we will find the optimum clustering
        for i in range(k - 1, nElems):
            # The mean and the variance of the cluster [j, i]
            mean = self.X[i]
            var = 0
            bestVar = listVar[i - 1] + var
            bestJ = i

            for j in reversed(range(k - 1, i)):
                newMean = ((i - j) * mean + self.X[j]) / (i - j + 1)
                var = var + (i - j) * (newMean - mean) ** 2 + (self.X[j] - newMean) ** 2
                if listVar[j - 1] + var < bestVar:
                    bestVar = listVar[j - 1] + var
                    bestJ = j
                mean = newMean

            listVarNew[i] = bestVar
            iniClustNew[i] = iniClust[bestJ - 1] + [bestJ]

        return listVarNew, iniClustNew

    def _calculateCenters(self, nClusters, iniClust):
        nElems = len(self.X)
        centers = np.zeros((nClusters, 1))
        for i in range(len(iniClust[-1]) - 1):
            iniPoint = iniClust[-1][i]
            endPoint = iniClust[-1][i + 1]
            centers[i, 0] = sum(self.X[iniPoint:endPoint]) / (endPoint - iniPoint)
        iniPoint = iniClust[-1][-1]
        centers[-1, 0] = sum(self.X[iniPoint:nElems]) / (nElems - iniPoint)