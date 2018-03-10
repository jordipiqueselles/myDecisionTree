import DecisionTree
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(True)
dt = DecisionTree.DecisionTree()
dt.fit(X, y)
pred = dt.predict(X)
print("Accuracy:", accuracy_score(y, pred))
print(list(y))
print(pred)
print(dt)
