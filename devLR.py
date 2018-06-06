import time
import numpy as np
import matplotlib.pyplot as plt
import logging as log
from sklearn.linear_model import LogisticRegression

## The function I will apply to z ##

def invCub(z, k=1):
    """
    It computes the inverse of f(z) = z**3 + k*z
    """
    auxU = lambda z, k: np.cbrt(np.sqrt(3) * np.sqrt(27 * z ** 2 + 4 * k ** 3) - 9 * z)
    u = auxU(z, k)
    return np.cbrt((2/3) * k) / u - u / (np.cbrt(2) * 3**(2/3))

def derInvCub(z, k=1):
    """
    It computes the derivative of the inverse of f(z) = z**3 + k*z. This derivative is 1/f'(z)
    :param z:
    :param k:
    :return:
    """
    return 1 / (3*z**2 + k)

#########################################

class MyLR:
    def __init__(self, fun=invCub, derFun=derInvCub, alfa=0.15, maxIter=1000, logLevel=log.DEBUG):
        self.fun = fun
        self.derFun = derFun
        self.alfa = alfa
        self.maxIter = maxIter
        self.B = None
        log.basicConfig(level=logLevel, format='%(asctime)s %(levelname)s %(message)s')

    def fit(self, X, y):
        nInst = len(X)
        nAttr = len(X[0])
        X = np.append(X, np.ones((nInst, 1)), axis=1)
        y = np.reshape(y, (nInst, 1))
        shuffledIdx = np.array(range(nInst))
        np.random.shuffle(shuffledIdx)
        X = X[shuffledIdx]
        y = y[shuffledIdx]
        # trampa #
        # self.B = (np.mean(X*y) - np.mean(X*(1-y), axis=0)).reshape((1,3))
        lr = LogisticRegression(C=99999999999)
        lr.fit(X, y.ravel())
        self.B = lr.coef_
        ##########
        # self.B = np.random.rand(1, nAttr + 1) * 2 - 1
        self._gradientDescent(X, y)

    def predict(self, X):
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        return (np.dot(X, self.B.transpose()) >= 0).flatten()

    def predict_proba(self, X):
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        Z = np.dot(X, self.B.transpose())
        funZ = self.fun(Z)
        prob = self._sigmoid(funZ)
        return prob

    def get_params(self, deep=True):
        return {"fun": self.fun, "derFun": self.derFun, "alfa": self.alfa, "maxIter": self.maxIter, "B": self.B}

    def _sigmoid(self, z):
        """
        The sigmoid function (1/(1+e^-z))
        """
        return 1 / (1 + np.exp(-z)) + 0.000001

    def _cost(self, X, y):
        """
        It calculates the cost of this Logistic Regression
        :param X: Matrix of instances (nRows = nInst; nCols = nAttr + 1). The last column is a ones column
        :param y: Column vector of classes (0, 1)
        :param B: Column vector of the parameters of the Logistic Regression
        """
        z = np.dot(X, self.B.transpose())
        funZ = self.fun(z)
        sig = self._sigmoid(funZ)
        return -np.dot(y.transpose(), np.log(sig)) - np.dot(1 - y.transpose(), np.log(1-sig))

    def _derCost(self, X, y):
        """
        It calculates the partial derivatives of the cost with respect to B
        :param X: Matrix of instances (nRows = nInst; nCols = nAttr + 1). The last column is a ones column
        :param y: Column vector of classes (0, 1)
        :param B: Column vector of the parameters of the Logistic Regression
        """
        dz = X
        z = np.dot(X, self.B.transpose())

        dFunZ = self.derFun(z)
        funZ = self.fun(z)

        sig = self._sigmoid(funZ)
        dsig = (-1)**y * sig * (1 - sig)
        sigAux = y*sig + (1-y)*(1-sig)

        dlog = 1 / sigAux

        der = dlog * dsig * dFunZ * dz
        return np.mean(der, axis=0)

    def _gradientDescent(self, X, y):
        log.info("Starting iterations")
        lastCost = np.inf
        derB = np.zeros(len(X[0]))
        for i in range(self.maxIter):
            # derB = self._derCost(X, y)
            # self.B = self.B - self.alfa * derB

            batchSize = 20
            beta = 0.2
            for batch in range(0, len(X), batchSize):
                auxDerB = self._derCost(X[batch:batch + batchSize], y[batch:batch + batchSize])
                derB = (1-beta)*derB + beta*auxDerB
                self.B = self.B - self.alfa*derB

            if i % 100 == 0:
                cost = self._cost(X, y)
                log.info("Iteration " + str(i))
                log.info("Cost " + str(cost))
                log.info("B " + str(self.B))
                log.info("derB " + str(derB))

                if (lastCost - cost) / cost < 0.05 or np.linalg.norm(derB) < 0.001:
                    log.info("Stopping at iteration " + str(i))
                    break
                lastCost = cost

def prova():
    # B -> Bi, Bi - 1, ..., B2, B1, B0
    # x -> xi, xi-1, ..., x2, x1, 1
    # X -> nRows = nInst; nCols = nAttr = i
    nInst = 1000
    nAttr = 2
    X0 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], nInst)
    X1 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], nInst)
    X = np.append(X0, X1, axis=0)
    y = np.array([[0] * nInst + [1] * nInst])

    lr = MyLR()
    lr.fit(X, y)

    # plt.scatter(X.transpose()[0], X.transpose()[1], c=y.flatten(), alpha=0.5, marker='o')
    # plt.show()

    # B = np.array([[1], [1], [0]])

if __name__ == '__main__':
    t = time.time()
    prova()
    print("Total time:", time.time() - t)

