import warnings

from sklearn.dummy import DummyClassifier

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import GridSearchCV, train_test_split
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
        estimator = DecisionTreeClassifier(random_state=101)

    elif type == 'r_forest':
        param_grid = [{'criterion': ['entropy', 'gini'],
                       'n_estimators': np.arange(5, 105
                                                 , 10),
                       'max_depth': range(2, 11),
                       'max_features': [None, 'sqrt', 'log2']
                       },
                      {'criterion': ['entropy', 'gini'],
                       'n_estimators': np.arange(5, 105, 10),
                       'min_samples_leaf': range(1, 10),
                       'max_features': [None, 'sqrt', 'log2']
                       }
        ]
        estimator = RandomForestClassifier(random_state=0)

    elif type == 'grad_boost':
        param_grid = {
            "loss": ["deviance", "exponential"],
            "learning_rate": [0.01, 0.1, 0.2],
            "min_samples_split": [2, 0.1],
            "max_depth": [5, 8],
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "squared_error"],
            "subsample": [0.5, 0.75, 1.0],
            "n_estimators": [10, 50, 100]
        }
        estimator = GradientBoostingClassifier(random_state=0)

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

    elif type == 'linear':
        param_grid = {'C': [0.1, 0.145, 0.15, 0.155, 0.2],
                      'penalty': ['l1', 'l2']
                      }
        estimator = LinearSVC()

    elif type == 'nu':
        param_grid = {'nu': [0.1, 0.235, 0.5, 0.75, 1],
                      'gamma': ['scale', 'auto', 10, 1, 0.1, 0.014678, 0.01, 0.001, 0.0001]}
        estimator = NuSVC()

    elif type == 'knn':
        param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
                      'weights': ['uniform', 'distance'],
                      'p': [1, 2, 3, 4, 5, 6]
                      }
        estimator = KNeighborsClassifier()

    else:
        {}
        estimator = None

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True, random_state=101)

    grid = GridSearchCV(estimator, param_grid=param_grid, cv=10, verbose=verbose, n_jobs=-1)

    grid.fit(X_train, y_train)

    return {'score': grid.best_score_, 'params': grid.best_params_, 'estimator': grid.best_estimator_}


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    full_data = dt.read_dataset()

    dt.remove_duplicates(full_data, same_label=False, inplace=True)
    # print(full_data.describe())
    # vl.correlation_matrix(data)

    # selected_genres = ['blues', 'classical', 'jazz', 'hiphop', 'pop', 'rock']
    # subset_data = full_data[full_data['genre'].isin(selected_genres)].copy()

    # Initialize Features and Target
    X, y = dt.extract_scaled_features_and_label(full_data)

    # Establish Train/Validation-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(X_train, y_train)
    print(knn_clf.score(X_test, y_test))

    reduced_data = full_data.drop('spec_cent', 1)

    # Initialize Features and Target
    X, y = dt.extract_scaled_features_and_label(reduced_data)

    # Establish Train/Validation-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(X_train, y_train)
    print(knn_clf.score(X_test, y_test))
    '''
    # Dummy Classifier
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    print(dummy.score(X_val, y_val))

    dummyf = DummyClassifier()
    dummyf.fit(Xf_train, yf_train)
    print(dummyf.score(Xf_val, yf_val))

    print()
    input()

    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(X_train, y_train)
    print(knn_clf.score(X_val, y_val))

    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(Xf_train, yf_train)
    print(knn_clf.score(Xf_val, yf_val))

    print()
    input()

    # Initialize Features and Target
    X, y = dt.extract_scaled_features_and_label(subset_data)

    Xf, yf = dt.extract_scaled_features_and_label(full_data)

    # Establish Train/Validation-Test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=101)
    Xf_train_val, Xf_test, yf_train_val, yf_test = train_test_split(Xf, yf, test_size=0.1, stratify=yf,
                                                                    random_state=101)

    # Traditional train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2,
                                                      stratify=y_train_val, random_state=101)
    Xf_train, Xf_val, yf_train, yf_val = train_test_split(Xf_train_val, yf_train_val, test_size=0.2,
                                                          stratify=yf_train_val, random_state=101)

    # Dummy Classifier
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    print(dummy.score(X_val, y_val))

    dummyf = DummyClassifier()
    dummyf.fit(Xf_train, yf_train)
    print(dummyf.score(Xf_val, yf_val))

    print()
    input()

    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(X_train, y_train)
    print(knn_clf.score(X_val, y_val))

    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(Xf_train, yf_train)
    print(knn_clf.score(Xf_val, yf_val))

    # Initialize Features and Target
    X, y = dt.extract_scaled_features_and_label(subset_data)

    Xf, yf = dt.extract_scaled_features_and_label(full_data)

    # Establish Train/Validation-Test split
    X_train_val, X_test, y_train_val, y_test = dt.tt_split(subset_data, scaled=True, random_state=101)
    Xf_train_val, Xf_test, yf_train_val, yf_test = dt.tt_split(full_data, scaled=True, random_state=101)

    # Dummy Classifier
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    print(dummy.score(X_val, y_val))

    dummyf = DummyClassifier()
    dummyf.fit(Xf_train, yf_train)
    print(dummyf.score(Xf_val, yf_val))

    print()
    input()

    # KNN
    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(X_train, y_train)
    print(knn_clf.score(X_val, y_val))

    knn_clf = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
    knn_clf.fit(Xf_train, yf_train)
    print(knn_clf.score(Xf_val, yf_val))
    '''
    '''
    train_scores = []
    scores = []
    for i in range(1, 21):
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True, random_state=101)

        clf = SVC(C=11, gamma=0.01468, kernel='rbf')
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))
        train_scores.append(clf.score(X_train, y_train))
        scores.append(clf.score(X_test, y_test))
        print("----------------\n")
    plt.plot(scores)
    plt.plot(train_scores)
    plt.show()
    '''
    '''[10]Decision Tree -> 0.620 {'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_leaf': 1}
    res = grid_search(data, type='dec_tree', verbose=2)
    dec_tree = res['estimator']
    cv_acc = res['score']
    dec_tree.fit(X_train, y_train)
    acc = dec_tree.score(X_test, y_test)
    print(f'Decision Tree -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''[9]Random Forest -> 0.788: {'criterion': 'entropy', 'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 65}
    res = grid_search(data, type='r_forest', verbose=2)
    dec_tree = res['estimator']
    cv_acc = res['score']
    print(dec_tree)
    dec_tree.fit(X_train, y_train)
    acc = dec_tree.score(X_test, y_test)
    print(f'Random Forest -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''[0]Gradient Boosting -> 0.776: {'criterion': 'squared_error', 'learning_rate': 0.1, 
                                       'loss': 'deviance', 'max_depth': 8, 'max_features': 'log2', 
                                       'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.75}
    res = grid_search(data, type='grad_boost', verbose=20)
    gb = res['estimator']
    cv_acc = res['score']
    print(gb)
    gb.fit(X_train, y_train)
    acc = gb.score(X_test, y_test)
    print(f'Gradient Boosting -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    # gb = GradientBoostingClassifier(random_state=0)
    # gb.fit(X_train, y_train)
    # print(gb.score(X_test, y_test))
    '''SVM -> 0.827 {'C': 11, 'gamma': 0.01468, 'kernel': 'rbf'}
    res = grid_search(data, type='svm')
    svm = res['estimator']
    cv_acc = res['score']
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    print(f'SVM -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''Linear SVM -> 0.765 {'C': 0.15, 'penalty': 'l2'}
    res = grid_search(data, type='linear')
    linSVM = res['estimator']
    cv_acc = res['score']
    linSVM.fit(X_train, y_train)
    acc = linSVM.score(X_test, y_test)
    print(f'LinearSVM -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''NuSVM -> 0.838 {'gamma': 'scale', 'nu': 0.235}
    res = grid_search(data, type='nu')
    nuSVM = res['estimator']
    cv_acc = res['score']
    nuSVM.fit(X_train, y_train)
    acc = nuSVM.score(X_test, y_test)
    print(f'NuSVM -> [M]{cv_acc} - {acc}: {res["params"]}')
    '''
    '''KNN -> 0.743 {'n_neighbors': 9, 'p': 2, 'weights': 'uniform'}
    res = grid_search(data, type='knn')
    knn = res['estimator']
    cv_acc = res['score']
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
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
