import dataset as dt

import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


def extract_features_and_label(data):
    '''Extract features and label from a dataset and return them separately as series

    :param data: dataset as a pd.DataFrame
    :return: features and associated label
    '''
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    return X, y


def extract_scaled_features_and_label(data):
    '''Scale and extract features and label from a dataset and return them separately as series

    :param data: dataset as a pd.DataFrame
    :return: scaled features and associated label
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data.drop(dt.LABEL, axis=1))
    scaled_features = scaler.transform(data.drop(dt.LABEL, axis=1))
    df_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])

    X = df_feat
    y = data[dt.LABEL]
    return X, y


def dec_tree(data, test_size=0.3):
    '''Apply a decision tree classifier to the dataset

    :param data: dataset as a pd.DataFrame
    :param test_size : float or int, default=None
        Parameter for the train_test_split function
    :return: accuracy of the model

    Examples
    --------
    >>> import features as ft
    >>> import dataset as dt
    >>> features = [ft.Features.CHROMAGRAM, ft.Features.MFCC]
    >>> n_mfcc = 15
    >>> data = dt.read_dataset(features, n_mfcc=n_mfcc)
    >>> accuracy = dec_tree(data)
    '''
    from sklearn.tree import DecisionTreeClassifier

    # Split dataset in features and target variable
    X, y = extract_features_and_label(data)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    return y_test, y_pred, metrics.accuracy_score(y_test, y_pred)


def knn(data, test_size=0.3, n_neighbors: int = 5):
    from sklearn.neighbors import KNeighborsClassifier

    # scale attribute values
    X, y = extract_scaled_features_and_label(data)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Create KNN classifer object
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train KNN Classifer
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)

    # print(metrics.confusion_matrix(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred))

    # Model Accuracy, how often is the classifier correct?
    return y_test, y_pred, metrics.accuracy_score(y_test, y_pred)


def elbow_method(data, test_size=0.3, max_k: int = 5):
    import matplotlib.pyplot as plt
    # use Elbow method to choose a correct k value for KNN
    error_rate = []
    opt_k = 1
    min_err = math.inf
    opt_test = []
    opt_pred = []
    opt_acc = 0
    for i in range(1, max_k):
        y_test, y_pred, acc = knn(data, test_size=test_size, n_neighbors=i)
        new_error = np.mean(y_pred != y_test)
        error_rate.append(new_error)
        if new_error < min_err:
            opt_k = i
            opt_test = y_test
            opt_pred = y_pred
            min_err = new_error
            opt_acc = acc

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
             markersize=10)
    plt.title('Error rate vs K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

    # print(f'{min_k}: Error[{min_err}] Accuracy[{min_acc}]')
    # print(metrics.confusion_matrix(min_test, min_pred))
    # print(metrics.classification_report(min_test, min_pred))
    return opt_k, opt_acc
