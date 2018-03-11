from DecisionTree import DecisionTree
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dt = DecisionTree(splitMethod=DecisionTree.splitStd)
dt.fit(X_train, y_train)
pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(list(y_test))
print(pred)
print(dt)
