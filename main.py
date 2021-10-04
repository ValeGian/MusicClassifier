import warnings

import visualize as vl
import features as ft
import classification as cl
import statistics as st
import dataset as dt
import optimize as opt

import librosa
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import cross_val_score

DEMONS = './demons_30.wav'


def test_dec_tree(data, criterion, n_iter: int = 10):
    accuracies = []

    # Create Decision Tree classifier object
    dtree = DecisionTreeClassifier(criterion=criterion)
    for i in range(n_iter):
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = dt.tt_split(data, randoma_state=None)

        # Train Decision Tree Classifier
        dtree.fit(X_train, y_train)

        # Model Accuracy, how often is the classifier correct?
        acc = dtree.score(X_test, y_test)

        accuracies.append(acc)
    accuracies = np.array(accuracies)
    return accuracies


def test_random_forest(data, criterion, n_estimators, n_iter: int = 10):
    accuracies = []

    # Create Random Forest classifier object
    rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    for i in range(n_iter):
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = dt.tt_split(data, randoma_state=None)

        # Train Random Forest Classifier
        rfc.fit(X_train, y_train)

        # Model Accuracy, how often is the classifier correct?
        acc = rfc.score(X_test, y_test)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    return accuracies


def test_elbow(data):
    opt_k = 1
    max_acc = 0
    for i in range(1, 10):
        k, acc = opt.elbow_method(data, max_k=20)
        if acc > max_acc:
            opt_k = k
            max_acc = acc
        print(acc)
    return opt_k, max_acc


def test_svm(data, type='def', kernel: str = 'rbf', C: any = 1.0, gamma: any = 'scale', penalty: any = 'l2',
             nu: any = 0.5):
    if type == 'linear':
        acc = cl.linearSVM(data, C=C, penalty=penalty)[-1]
    elif type == 'nu':
        acc = cl.nuSVM(data, nu=nu, gamma=gamma)[-1]
    else:
        acc = cl.svm(data, kernel=kernel, C=C, gamma=gamma)[-1]

    return acc


def grid_search(data, type='default'):
    from sklearn.model_selection import GridSearchCV

    if type == 'svm':
        param_grid = [{'kernel': ['rbf'],
                       'C': [0.1, 1, 10, 100, 1000],
                       'gamma': ['scale', 'auto', 1, 1e-1, 1e-2, 1e-3, 1e-4]
                       }
                       # 'rbf' ALWAYS results in better accuracy
                       # ,
                       # {'kernel': ['linear'],
                       # 'C': [0.1, 1, 10, 100, 1000]
                       # },
                       # {'kernel': ['poly'],
                       # 'degree': [1, 2, 3, 4, 5, 6],
                       # 'gamma': ['scale', 'auto', 1, 1e-1, 1e-2, 1e-3, 1e-4]
                       # }
                      ]
        estimator = SVC()
        scale = True

    elif type == 'linear':
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'penalty': ['l1', 'l2']
                      }
        estimator = LinearSVC()
        scale = True

    elif type == 'nu':
        param_grid = {'nu': [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1],
                      'gamma': ['scale', 'auto', 1, 1e-1, 1e-2, 1e-3, 1e-4]}
        estimator = NuSVC()
        scale = True

    elif type == 'knn':
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
                      'weights': ['uniform', 'distance'],
                      'p': [1, 2, 3, 4, 5, 6]
                      }
        estimator = KNeighborsClassifier()
        scale = True

    else:
        {}
        estimator = None
        scale = False

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=scale, randoma_state=101)

    grid = GridSearchCV(estimator, param_grid=param_grid, cv=10, scoring='accuracy', verbose=0, n_jobs=None)

    grid.fit(X_train, y_train)

    return {'score': grid.best_score_, 'params': grid.best_params_, 'estimator': grid.best_estimator_}


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    test_num = 100
    labels = dt.get_labels()
    data = dt.read_dataset(labels=['blues', 'classical', 'jazz', 'metal', 'pop', 'rock'], n_mfcc=20)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=False, randoma_state=101)
    sX_train, sX_test, sy_train, sy_test = dt.tt_split(data, scaled=True, randoma_state=101)

    '''SVM -> 0.861 {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
    res = grid_search(data, type='svm')
    svm = res['estimator']
    svm.fit(sX_train, sy_train)
    acc = svm.score(sX_test, sy_test)
    print(f'SVM -> {acc}: {res["params"]}')
    '''
    '''Linear SVM -> 0.777 {'C': 1, 'penalty': 'l2'}
    res = grid_search(data, type='linear')
    linSVM = res['estimator']
    linSVM.fit(sX_train, sy_train)
    acc = linSVM.score(sX_test, sy_test)
    print(f'LinearSVM -> {acc}: {res["params"]}')
    '''
    '''NuSVM -> 0.872 {'gamma': 'scale', 'nu': 0.01}
    res = grid_search(data, type='nu')
    nuSVM = res['estimator']
    nuSVM.fit(sX_train, sy_train)
    acc = nuSVM.score(sX_test, sy_test)
    print(f'NuSVM -> {acc}: {res["params"]}')
    '''
    '''KNN -> 0.805 {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
    res = grid_search(data, type='knn')
    knn = res['estimator']
    knn.fit(sX_train, sy_train)
    acc = knn.score(sX_test, sy_test)
    print(f'KNN -> {acc}: {res["params"]}')
    '''
