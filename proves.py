from DecisionTree import DecisionTree
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import time
from scipy.io import arff
import os
import numpy as np
import pandas as pd


def loadCsv(path):
    df = pd.read_csv(path, sep=",", header=None, na_values='?')
    df = df.dropna()
    X = df.iloc[:,:-1].values.tolist()
    y = df.iloc[:,-1].values.tolist()
    return X, y


def loadArff(path):
    data, metadata = arff.loadarff(path)
    df = pd.DataFrame(data)
    df = df.dropna()
    numericCols = [colType == 'numeric' for colType in metadata.types()]
    X = df.iloc[:,numericCols].values.tolist()
    if 'class' in df.columns:
        y = df.loc[:,'class'].values.tolist()
    elif 'Class' in df.columns:
        y = df.loc[:, 'Class'].values.tolist()
    else:
        raise AttributeError("Class not found for " + path)
    return X, y


def readFile(path):
    print("Reading ", path)
    if path[-4:] == '.csv':
        X, y = loadCsv(path)
    elif path[-5:] == '.arff':
        X, y = loadArff(path)
    else:
        raise ValueError("Invalid extension for: " + path)
    return X, y


def writeResultsCsv(dataset, nInst, nAttr, bestPrior, method, acc, auc, t, numNodes):
    with open(nameOutFile, 'a') as f:
        f.write(dataset + ";" + str(nInst) + ";" + str(nAttr) + ";" + str(bestPrior) + ";" + method + ";" +
                str(acc) + ';' + str(auc) + ';' + str(t) + ';' + str(numNodes) + '\n')


def evalFolderDatasets():
    # folder = "C:\\Users\\pique\\OneDrive - HP Inc\\MIRI\\FastRandomForest\\datasets\\RDG1_generator\\"
    # folder = "C:\\Users\\pique\\OneDrive - HP Inc\\MIRI\\FastRandomForest\\datasets\\real_datasets\\"
    folder = ".\\datasets\\"
    # take only .arff or .csv files
    listFileName = [fileName for fileName in os.listdir(folder) if fileName[-5:] == ".arff" or fileName[-4:] == '.csv']
    # listSplitMethods = [DecisionTree.splitStd, DecisionTree.splitLR, DecisionTree.splitKmeans]
    listSplitMethods = [DecisionTree.splitKmeans]

    for fileName in listFileName[:]:
        X, y = readFile(folder + fileName)

        # meta info about the dataset (numInstances, numAttributes, nClasses...)
        nClasses = set(y)
        # Only the datasets that have less than 100 attributes and have extactly 2 classes will be used
        if len(X) < 30 or len(X[0]) > 100 or len(nClasses) != 2:
            print("################################################################")
            print()
            continue

        distrClss = [y.count(elem) for elem in set(y)]
        bestPrior = max(distrClss) / sum(distrClss)

        print("Learning ", fileName)
        print("nInst:", len(X), "| nAttr:", len(X[0]), "| nClasses:", len(nClasses), "| distrClasses:", distrClss, "| bestPrior:", bestPrior)
        print()

        # Train and test split
        np.random.seed(2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        for method in listSplitMethods:
            print("---------------------------------------------------------------")
            print("Method:", method)

            if False:
                # Evaluate the classifier using 10-fold cross-validation
                t = time.time()
                dt = DecisionTree(splitMethodName=method, maxDepth=2)
                scores = cross_val_score(dt, X, y, cv=8, n_jobs=1)
                t = time.time() - t
                acc = scores.mean()
                print("CV accuracy:", acc)
                print("Time:", t)
                print()

                writeResultsCsv(fileName, len(X), len(X[0]), bestPrior, method, acc, 0, t, 0)

            else:
                # Only to evaluate the first split
                t = time.time()
                dt = DecisionTree(splitMethodName=method, maxDepth=0)
                # dt = DecisionTreeClassifier()
                dt.fit(X_train, y_train)
                dt.node.plotSplit()
                pred = dt.predict(X_test)
                acc = accuracy_score(y_test, pred)
                prob = [pr[1] for pr in dt.predict_proba(X_test)]
                # prob = dt.predict_proba(X_test)
                y_test_bin = dt._transClassToIdx(y_test)
                auc = roc_auc_score(y_test_bin, prob)
                t = time.time() - t
                print("Accuracy:", acc)
                print("AUC:", auc)
                print("Time:", t)
                print()
                print(dt)

                writeResultsCsv(fileName, len(X), len(X[0]), bestPrior, method, acc, auc, t, dt.node.getNumNodes())

        print("################################################################")
        print()


def testClassifierCv():
    X, y = load_breast_cancer(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("nInst:", len(X), "| nAttr:", len(X[0]), "| nClasses:", len(set(y)))
    t = time.time()
    dt = DecisionTree(splitMethodName=DecisionTree.splitStd)
    scores = cross_val_score(dt, X, y, cv=8, n_jobs=1)
    t = time.time() - t
    print("CV accuracy:", scores.mean())
    print("Time:", t)


def testClassifierTrainTest(splitMethod):
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("nInst:", len(X), "| nAttr:", len(X[0]), "| nClasses:", len(set(y)))
    t = time.time()
    dt = DecisionTree(splitMethodName=splitMethod)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    acc = accuracy_score(y_test, pred)
    t = time.time() - t
    print("Accuracy:", acc)
    print("Time:", t)
    print()
    print(dt)


if __name__ == '__main__':
    nameOutFile = "provesOut.csv"
    seed = 2
    # np.random.seed(seed)
    # print("Split LR")
    # testClassifierTrainTest(DecisionTree.splitLR)
    # np.random.seed(seed)
    # print("Split Std")
    # testClassifierTrainTest(DecisionTree.splitStd)

    np.random.seed(seed)
    evalFolderDatasets()

    # dt.fit(X_train, y_train)
    # pred = dt.predict(X_test)
    # print("Accuracy:", accuracy_score(y_test, pred))
    # print(list(y_test))
    # print(pred)
    # print(dt)
