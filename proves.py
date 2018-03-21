from DecisionTree import DecisionTree
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import time
from scipy.io import arff
import os


def preprocessData(data, metadata):
    X = []
    for row in data:
        newRow = []
        for elem in list(row)[:-1]:
            if elem == b'true':
                newRow.append(1)
            elif elem == b'false':
                newRow.append(0)
            else:
                newRow.append(elem)
        X.append(newRow)

    yRaw = [row[-1] for row in data]
    classVals = dict((clss, idx) for (idx, clss) in enumerate(set(yRaw)))
    y = [classVals[clss] for clss in yRaw]
    return X, y

def evalFolderDatasets():
    folder = "C:\\Users\\pique\\OneDrive - HP Inc\\MIRI\\FastRandomForest\\datasets\\RDG1_generator\\"
    listFileName = [fileName for fileName in os.listdir(folder) if fileName[-5:] == ".arff"]
    listSplitMethods = [DecisionTree.splitStd, DecisionTree.splitPca]
    for fileName in listFileName:
        data, metadata = arff.loadarff(folder + fileName)
        X, y = preprocessData(data, metadata)
        if len(X) > 800 or len(X[0]) > 800:
            continue
        print(fileName)
        print("nInst:", len(X), "| nAttr:", len(X[0]), "| nClasses:", len(set(y)))
        print()

        for method in listSplitMethods:
            print("Method:", method)
            t = time.time()
            dt = DecisionTree(splitMethodName=method)
            scores = cross_val_score(dt, X, y, cv=8, n_jobs=-1)
            t = time.time() - t
            print("CV accuracy:", scores.mean())
            print("Time:", t)
            print()

        print("---------------------------------------")
        print()


def testClassifier():
    X, y = load_breast_cancer(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("nInst:", len(X), "| nAttr:", len(X[0]), "| nClasses:", len(set(y)))
    t = time.time()
    dt = DecisionTree(splitMethodName=DecisionTree.splitPca)
    scores = cross_val_score(dt, X, y, cv=8, n_jobs=1)
    t = time.time() - t
    print("CV accuracy:", scores.mean())
    print("Time:", t)


if __name__ == '__main__':
    # evalFolderDatasets()
    testClassifier()

    # dt.fit(X_train, y_train)
    # pred = dt.predict(X_test)
    # print("Accuracy:", accuracy_score(y_test, pred))
    # print(list(y_test))
    # print(pred)
    # print(dt)
