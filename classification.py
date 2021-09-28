import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

import features

def dec_tree():
    col_names = []
    for feature in features.FEATURES_TO_CONSIDER:
        if feature != features.Features.MFCC:
            col_names.append(feature.name.lower())
        else:
            for i in range(features.DEF_N_MFCC):
                col_names.append(f'{features.Features.MFCC.name.lower()}{i}')
    col_names.append('label')

    data = pd.read_csv("data.csv", header=None, names=col_names)
    data = data.drop(labels=0, axis=0) # drop row 0 of the dataframe

    # split dataset in features and target variable
    feature_cols = col_names
    feature_cols.pop(-1)
    X = data[feature_cols]
    y = data['label']

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))