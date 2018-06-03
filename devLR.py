import numpy as np
import matplotlib.pyplot as plt

k = 1

def invCub(z, k):
    auxU = lambda z, k: np.cbrt(np.sqrt(3) * np.sqrt(27 * z ** 2 + 4 * k ** 3) - 9 * z)
    u = auxU(z, k)
    return np.cbrt((2/3) * k) / u - u / (np.cbrt(2) * 3**(2/3))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, B):
    z = np.dot(X, B.transpose())
    cubR = invCub(z, k)
    sig = sigmoid(cubR)
    return -np.dot(y, np.log(sig)) + np.dot(1 - y, np.log(1-sig))

def derCost(X, y, B):
    y = y.transpose()
    dz = X
    z = np.dot(X, B.transpose())
    dCubR = 1 / (3*z**2 + k)
    cubR = invCub(z, k)
    sig = sigmoid(cubR)
    dsig = (-1)**y * sig * (1 - sig)
    sigAux = y*sig + (1-y)*(1-sig)
    dlog = 1 / sigAux
    der = dlog * dsig * dCubR * dz
    return np.mean(der, axis=0)

# B -> Bi, Bi-1, ..., B2, B1, B0
# x -> xi, xi-1, ..., x2, x1, 1
# X -> nRows = nInst; nCols = nAttr = i
nInst = 1000
nAttr = 2
X0 = np.random.multivariate_normal([-2,-2], [[1,0],[0,1]], nInst)
X1 = np.random.multivariate_normal([2,2], [[1,0],[0,1]], nInst)
X = np.append(X0, X1, axis=0)
X = np.append(X, np.ones((nInst*2, 1)), axis=1)
y = np.array([[0]*nInst + [1]*nInst])

# plt.scatter(X.transpose()[0], X.transpose()[1], c=y.flatten(), alpha=0.5, marker='o')
# plt.show()

B = np.random.rand(1, nAttr+1)*2 - 1
# B = np.array([[1], [1], [0]])

print(B)
print(cost(X, y, B))
print(derCost(X, y, B))

print("Starting iterations")
for i in range(1000):
    derB = derCost(X, y, B)
    B = B - 0.1*derB
    if i % 10 == 0:
        print("Iteration", i)
        print(cost(X, y, B))
        print(B)
        print(derB)

