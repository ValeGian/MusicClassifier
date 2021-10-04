import math
import numpy as np
import matplotlib.pyplot as plt

import dataset as dt


def elbow_method(data, max_k: int = 5, weights: any = 'uniform', p: int = 2, verbose=False):
    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier

    # use Elbow method to choose a correct k value for KNN
    error_rate = []
    opt_k = 1
    min_err = math.inf
    opt_test = []
    opt_pred = []
    opt_acc = 0
    for i in range(1, max_k):
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True, randoma_state=None)

        # Create KNN classifier object
        knn = KNeighborsClassifier(n_neighbors=i, weights=weights, metric='minkowski', p=p)

        # Train KNN Classifier
        knn.fit(X_train, y_train)

        # Predict the response for test dataset
        y_pred = knn.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        acc = metrics.accuracy_score(y_test, y_pred)

        new_error = np.mean(y_pred != y_test)
        error_rate.append(new_error)
        if new_error < min_err:
            opt_k = i
            opt_test = y_test
            opt_pred = y_pred
            min_err = new_error
            opt_acc = acc

    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
                 markersize=10)
        plt.title('Error rate vs K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()

    return opt_k, opt_acc
