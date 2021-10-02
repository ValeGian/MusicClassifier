import warnings

import visualize as vl
import features as ft
import classification as cl
import statistics as st
import dataset as dt

import librosa
import numpy as np
import os

DEMONS = './demons_30.wav'


def test_dec_tree(data, criterion, n_iter: int = 10):
    accuracies = []
    for i in range(n_iter):
        y_test, y_pred, acc = cl.dec_tree(data, criterion=criterion)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    return accuracies


def test_elbow(data):
    opt_k = 1
    max_acc = 0
    for i in range(1, 10):
        k, acc = cl.elbow_method(data, max_k=20)
        if acc > max_acc:
            opt_k = k
            max_acc = acc
    return opt_k, max_acc


def test_svm(data, type='def', C: any = 1.0, gamma: any = 'scale', penalty: any = 'l2', nu: any = 0.5):
    if type == 'linear':
        acc = cl.linearSVM(data, C=C, penalty=penalty)[-1]
    elif type == 'nu':
        acc = cl.nuSVM(data, nu=nu, gamma=gamma)[-1]
    else:
        acc = cl.svm(data, C=C, gamma=gamma)[-1]

    return acc


def test_grid_search(data, type='def'):
    from sklearn.svm import SVC, LinearSVC, NuSVC
    # Split dataset in features and target variable
    X, y = cl.extract_scaled_features_and_label(data)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = cl.train_test_split(X, y, test_size=0.3, random_state=101)

    if type == 'linear':
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
        estimator = LinearSVC()
    elif type == 'nu':
        param_grid = {'nu': [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1], 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001]}
        estimator = NuSVC()
    else:
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001]}
        estimator = SVC()

    best_params = cl.grid_search(param_grid, X_train=X_train, y_train=y_train, estimator=estimator)
    return best_params


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    test_num = 100
    labels = dt.get_labels()
    data = dt.read_dataset(labels=['blues', 'classical', 'jazz', 'metal', 'pop', 'rock'])

    '''
    # TEST dec tree 'gini' -> 0.693
    accuracies = test_dec_tree(data, criterion='gini', n_iter=test_num)
    avg = np.mean(accuracies)
    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(f'Decision Tree with gini AVG: {avg} -> {conf_int}\n')
    
    # print(metrics.confusion_matrix(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred))
    
    # TEST dec tree 'entropy' -> 0.698
    accuracies = test_dec_tree(data, criterion='entropy', n_iter=test_num)
    avg = np.mean(accuracies)
    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(f'Decision Tree with entropy AVG: {avg} -> {conf_int}\n')

    #TEST elbow method for KNN -> 0.855
    opt_k = 1
    max_acc = 0
    for i in range(test_num):
        k, acc = test_elbow(data)
        print(i)
        if acc > max_acc:
            opt_k = k
            max_acc = acc
    print(f'KNN  -> {max_acc}')
    '''

    # TEST SVM -> 0.866
    best_acc = test_svm(data, C=100, gamma='scale')
    print(f'SVM -> {best_acc}')

    # TEST LINEAR SVM -> 0.788
    best_acc = test_svm(data, 'linear', C=10, penalty='l2')
    print(f'Linear SVM -> {best_acc}')

    # TEST NuSVM -> 0.861
    best_acc = test_svm(data, 'nu', gamma='scale', nu=0.1)
    print(f'Nu SVM -> {best_acc}')

    '''
    def_best_params = test_grid_search(data)

    lin_best_params = test_grid_search(data, 'linear')

    nu_best_params = test_grid_search(data, 'nu')

    print(f'{def_best_params}\n{lin_best_params}\n{nu_best_params}')
    '''
