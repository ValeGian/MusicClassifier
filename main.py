import warnings

import visualize as vl
import features as ft
import classification as cl
import statistics as st
import dataset as dt
import optimize as opt

import librosa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

DEMONS = './demons_30.wav'


def test_dec_tree(data, criterion, n_iter: int = 10):
    accuracies = []

    # Create Decision Tree classifier object
    dtree = DecisionTreeClassifier(criterion=criterion)
    for i in range(n_iter):
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = dt.tt_split(data, random_state=None)

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
        X_train, X_test, y_train, y_test = dt.tt_split(data, random_state=None)

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


def grid_search(data, type='default', verbose=0):

    if type == 'dec_tree':
        param_grid = [{'criterion': ['entropy', 'gini'],
                       'max_depth': range(2, 11),
                       'max_features': [None, 'sqrt', 'log2']
                       },
                      {'criterion': ['entropy', 'gini'],
                       'min_samples_leaf': range(1, 10),
                       'max_features': [None, 'sqrt', 'log2']
                       }
        ]
        estimator = DecisionTreeClassifier(random_state=10)
        scale = True

    elif type == 'r_forest':
        x = 2
        estimator = RandomForestClassifier()
        scale = False

    elif type == 'svm':
        param_grid = [{'kernel': ['rbf'], # 'rbf' ALWAYS results in better accuracy
                       'C': [0.1, 1, 11, 100, 1000],
                       'gamma': ['scale', 'auto', 10, 1, 0.1, 0.014678, 0.01, 0.001, 0.0001]
                       }
                        ,
                        #{'kernel': ['linear'], # 0.810 {'C': 0.15, 'kernel': 'linear'}
                        #'C': [0.145, 0.148, 0.15, 0.152, 0.155]
                        #}
                        #,
                        #{'kernel': ['poly'], # 0.815 {'degree': 1, 'gamma': 0.1, 'kernel': 'poly'}
                        #'degree': [1, 2, 3, 4, 5],
                        #'gamma': ['scale', 'auto', 1e1, 2, 1.1, 1, 0.9, 0.1, 0.01]
                        #}
        ]
        estimator = SVC()
        scale = True

    elif type == 'linear':
        param_grid = {'C': [0.1, 0.145, 0.15, 0.155, 0.2],
                      'penalty': ['l1', 'l2']
                      }
        estimator = LinearSVC()
        scale = True

    elif type == 'nu':
        param_grid = {'nu': [0.1, 0.235, 0.5, 0.75, 1],
                      'gamma': ['scale', 'auto', 10, 1, 0.1, 0.014678, 0.01, 0.001, 0.0001]}
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
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=scale, random_state=101)

    grid = GridSearchCV(estimator, param_grid=param_grid, cv=10, verbose=verbose, n_jobs=None)

    grid.fit(X_train, y_train)

    return {'score': grid.best_score_, 'params': grid.best_params_, 'estimator': grid.best_estimator_}


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    test_num = 100
    labels = dt.get_labels()
    data = dt.read_dataset(labels=['blues', 'classical', 'jazz', 'hiphop', 'pop', 'rock'], n_mfcc=20)
    dt.remove_duplicates(data, same_label=False, inplace=True)
    print(data['label'].value_counts())

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=False, random_state=101)
    sX_train, sX_test, sy_train, sy_test = dt.tt_split(data, scaled=True, random_state=101)

    '''Decision Tree -> 0.620 {'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_leaf': 1}'''
    res = grid_search(data, type='dec_tree', verbose=2)
    dec_tree = res['estimator']
    cv_acc = res['score']
    dec_tree.fit(X_train, y_train)
    acc = dec_tree.score(X_test, y_test)
    print(f'Decision Tree -> [M]{cv_acc} - {acc}: {res["params"]}')
    ''''''
    '''SVM -> 0.827 {'C': 11, 'gamma': 0.01468, 'kernel': 'rbf'}
    res = grid_search(data, type='svm')
    svm = res['estimator']
    cv_acc = res['score']
    svm.fit(sX_train, sy_train)
    acc = svm.score(sX_test, sy_test)
    print(f'SVM -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''Linear SVM -> 0.765 {'C': 0.15, 'penalty': 'l2'}
    res = grid_search(data, type='linear')
    linSVM = res['estimator']
    cv_acc = res['score']
    linSVM.fit(sX_train, sy_train)
    acc = linSVM.score(sX_test, sy_test)
    print(f'LinearSVM -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''NuSVM -> 0.838 {'gamma': 'scale', 'nu': 0.235}
    res = grid_search(data, type='nu')
    nuSVM = res['estimator']
    cv_acc = res['score']
    nuSVM.fit(sX_train, sy_train)
    acc = nuSVM.score(sX_test, sy_test)
    print(f'NuSVM -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''KNN -> 0.743 {'n_neighbors': 9, 'p': 2, 'weights': 'uniform'}
    res = grid_search(data, type='knn')
    knn = res['estimator']
    cv_acc = res['score']
    knn.fit(sX_train, sy_train)
    acc = knn.score(sX_test, sy_test)
    print(f'KNN -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''

    '''PCA
    pca = PCA()
    svm = SVC()
    pipe = Pipeline(steps=[('pca', pca),
                           ('svm', svm)])
    n_components = list(range(1, data.shape[1] + 1, 1))
    C = [0.1, 1, 11, 100, 1000]
    gamma = ['scale', 'auto', 10, 1, 0.1, 0.014678, 0.01, 0.001, 0.0001]
    parameters = dict(pca__n_components=n_components,
                      svm__gamma=gamma,
                      svm__C=C)
    clf_GS = GridSearchCV(pipe, parameters, verbose=2)
    clf_GS.fit(X_train, y_train)

    print('Best gamma:', clf_GS.best_estimator_.get_params()['svm__gamma'])
    print('Best C:', clf_GS.best_estimator_.get_params()['svm__C'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print()
    print(clf_GS.best_estimator_.get_params()['svm'])
    svm = clf_GS.best_estimator_.get_params()['svm']
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    print(f'SVM -> {acc}: {clf_GS.best_params_}')
    '''
