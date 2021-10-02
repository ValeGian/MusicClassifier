import math

import numpy as np

import classification as cl


def elbow_method(data, max_k: int = 5):
    # use Elbow method to choose a correct k value for KNN
    error_rate = []
    opt_k = 1
    min_err = math.inf
    opt_test = []
    opt_pred = []
    opt_acc = 0
    for i in range(1, max_k):
        y_test, y_pred, acc = cl.knn(data, n_neighbors=i)
        new_error = np.mean(y_pred != y_test)
        error_rate.append(new_error)
        if new_error < min_err:
            opt_k = i
            opt_test = y_test
            opt_pred = y_pred
            min_err = new_error
            opt_acc = acc
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
             markersize=10)
    plt.title('Error rate vs K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()
    '''
    return opt_k, opt_acc


def grid_search(param_grid, X_train, y_train, estimator):
    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(estimator, param_grid=param_grid, verbose=3)

    grid.fit(X_train, y_train)

    return grid.best_params_
