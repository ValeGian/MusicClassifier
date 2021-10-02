from sklearn import metrics

import dataset as dt


################
# DECISION TREE
def dec_tree(data, criterion: any = 'gini'):
    """Apply a decision tree classifier to the dataset

    :param data: dataset as a pd.DataFrame
    :param criterion: {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    :return: accuracy of the model

    Examples
    --------
    >>> import features as ft
    >>> import dataset as dt
    >>> features = [ft.Features.CHROMAGRAM, ft.Features.MFCC]
    >>> n_mfcc = 15
    >>> data = dt.read_dataset(features, n_mfcc=n_mfcc)
    >>> accuracy = dec_tree(data)
    """
    from sklearn.tree import DecisionTreeClassifier

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data)

    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier(criterion=criterion)

    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)
    return y_test, y_pred, acc


######
# KNN
def knn(data, n_neighbors: int = 5):
    from sklearn.neighbors import KNeighborsClassifier

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True)

    # Create KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train KNN Classifier
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)

    return y_test, y_pred, acc


######
# SVM
def svm(data, C: any = 1.0, gamma: any = 'scale'):
    from sklearn.svm import SVC

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True)

    # Create SVM classifier object
    svm = SVC(C=C, gamma=gamma)

    # Train SVM Classifier
    svm.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = svm.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)

    return y_test, y_pred, acc


def linearSVM(data, C: any = 1.0, penalty: any = 'l2'):
    from sklearn.svm import LinearSVC

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True)

    # Create SVM classifier object
    svm = LinearSVC(C=C, penalty=penalty)

    # Train SVM Classifier
    svm.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = svm.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)

    return y_test, y_pred, acc


def nuSVM(data, nu: any = 0.5, gamma: any = 'scale'):
    from sklearn.svm import NuSVC

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = dt.tt_split(data, scaled=True)

    # Create SVM classifier object
    svm = NuSVC(nu=nu, gamma=gamma)

    # Train SVM Classifier
    svm.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = svm.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)

    return y_test, y_pred, acc
