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

DEMONS = './demons_30.wav'


def test_dec_tree(data, criterion, n_iter: int = 10):
    accuracies = []
    for i in range(n_iter):
        y_test, y_pred, acc = cl.dec_tree(data, criterion=criterion)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    return accuracies


def test_random_forest(data, criterion, n_estimators, n_iter: int = 10):
    accuracies = []
    for i in range(n_iter):
        y_test, y_pred, acc = cl.random_forest(data, n_estimators=n_estimators, criterion=criterion)
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
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.neighbors import KNeighborsClassifier

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
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=scale)

    grid = opt.grid_search(param_grid, X_train=X_train, y_train=y_train, estimator=estimator)
    return {'score': grid.best_score_, 'params': grid.best_params_}


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    test_num = 100
    labels = dt.get_labels()
    data = dt.read_dataset(labels=['blues', 'classical', 'jazz', 'metal', 'pop', 'rock'], n_mfcc=17)
    '''
    # TEST dec tree 'gini' -> 0.693
    accuracies = test_dec_tree(data, criterion='gini', n_iter=test_num)
    avg = np.mean(accuracies)
    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(f'Decision Tree with gini AVG: {avg} -> {conf_int}\n')
    
    # TEST dec tree 'entropy' -> 0.698
    accuracies = test_dec_tree(data, criterion='entropy', n_iter=test_num)
    avg = np.mean(accuracies)
    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(f'Decision Tree with entropy AVG: {avg} -> {conf_int}\n')

    # TEST random forest 'gini -> 0.829
    accuracies = test_random_forest(data, n_estimators=100, criterion='gini', n_iter=test_num)
    avg = np.mean(accuracies)
    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(f'Random Forest with gini AVG: {avg} -> {conf_int}\n')
    
    # TEST random forest 'entropy -> 0.824
    accuracies = test_random_forest(data, n_estimators=100, criterion='entropy', n_iter=test_num)
    avg = np.mean(accuracies)
    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(f'Random Forest with entropy AVG: {avg} -> {conf_int}\n')
    '''
    # print(grid_search(data, type='knn'))
    # print(cl.knn(data, n_neighbors=5, weights='distance', p=1)[-1])
    '''
    #TEST elbow method for KNN -> 0.855
    opt_k = 1
    max_acc = 0
    for i in range(10):
        k, acc = test_elbow(data)
        print(i)
        if acc > max_acc:
            opt_k = k
            max_acc = acc
    print(f'KNN  -> {max_acc}')
    '''
    '''
    # TEST SVM -> 0.872
    best_acc = test_svm(data, kernel='rbf', C=10, gamma='scale')
    print(f'SVM -> {best_acc}')

    # TEST LINEAR SVM -> 0.766
    # best_acc = test_svm(data, 'linear', C=10, penalty='l2')
    # print(f'Linear SVM -> {best_acc}')

    # TEST NuSVM -> 0.894
    best_acc = test_svm(data, 'nu', gamma='scale', nu=0.1)
    print(f'Nu SVM -> {best_acc}')
    '''
    '''
    def_best_params = grid_search(data, type='svm')

    # lin_best_params = grid_search(data, type='linear') -> WORST OF THE 3

    nu_best_params = grid_search(data, type='nu')

    print(f'{def_best_params}\n{nu_best_params}')
    '''
    for i in range(13, 21):
        data = dt.read_dataset(labels=['blues', 'classical', 'jazz', 'metal', 'pop', 'rock'], n_mfcc=i)
        def_best_params = grid_search(data, type='svm')

        print(f'{i}: {def_best_params}')
