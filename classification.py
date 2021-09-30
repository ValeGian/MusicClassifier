import features as ft

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def dec_tree(data, test_size=None):
    '''Apply a decision tree classifier to the dataset

    :param data: dataset as a pd.DataFrame
    :param test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
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

    # split dataset in features and target variable
    feature_cols = data.columns.tolist()
    label = feature_cols.pop(-1)    # delete 'label'
    X = data[feature_cols]
    y = data[label]

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    return metrics.accuracy_score(y_test, y_pred)
