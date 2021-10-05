import features as ft

import os
import csv
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = './dataset'
DATASET = 'data.csv'
LABEL = 'label'


def write_dataset(header: list[str]):
    """Extract features from the audio signals in the original dataset

    In output we will have the data.csv file, that will be our dataset

    :param header: list of feature names
    :return:

    Examples
    --------
    >>> import features as ft
    >>> features = [ft.Features.CHROMAGRAM, ft.Features.MFCC]
    >>> n_mfcc = 15
    >>> exp_features = ft.explode_features(features, n_mfcc=n_mfcc)
    >>> write_dataset(exp_features)
    """

    with open(DATASET, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # add a row for each song in the original dataset
        count = 0
        for genre in os.listdir(DATASET_PATH):
            print(genre)
            for filename in os.listdir(f'{DATASET_PATH}/{genre}'):
                song_path = f'{DATASET_PATH}/{genre}/{filename}'
                x, sr = librosa.load(song_path)
                features = ft.extract_features(x, sr=sr)
                features.append(genre)  # add the label

                # Insert the new calculated values of this song in the file
                writer.writerow(features)

                count += 1
                print(count)


def read_dataset(features: list[ft.Features] = ft.DEF_FEATURES, n_mfcc: int = ft.DEF_N_MFCC, labels: list[str] = []):
    """Read specified features from dataset

    :param features: list of features to read
    :param n_mfcc: number of MFCC features
    :param labels: list of class names
            Used to select specific rows. [] = select every class
    :return: dataset as a pd.DataFrame
    """
    names = ft.explode_features(ft.DEF_FEATURES, n_mfcc=ft.DEF_N_MFCC)
    selected_cols = ft.explode_features(features, n_mfcc=n_mfcc)
    data = pd.read_csv(DATASET, header=None, skipinitialspace=True, names=names, usecols=selected_cols)
    if labels:
        data = data[data[LABEL].isin(labels)]
    else:
        data = data.drop(labels=0, axis=0)  # drop row 0 of the dataframe
    return data


def remove_duplicates(data, same_label=False, inplace=False):
    """Remove duplicate song

    :param data: dataset as a pd.DataFrame
    :param same_label: bool, default False
            If True, remove only equal songs which have the same label
    :param inplace: bool, default False
            Whether to drop duplicates in place or to return a copy
    :return: dataset without duplicate
    """
    columns = data.columns.values.tolist()
    # Drop duplicates with same genre (label), keeping only the first occurrence
    if inplace:
        data.drop_duplicates(subset=columns, keep='first', inplace=True)
    else:
        new_df = data.drop_duplicates(subset=columns, keep='first')

    if same_label:
        if inplace:
            return data
        return new_df

    columns.remove(LABEL)
    # Drop all duplicates
    if inplace:
        data.drop_duplicates(subset=columns, keep=False, inplace=True)
        return data

    new_df.drop_duplicates(subset=columns, keep=False, inplace=True)
    return new_df


def get_labels():
    """Get label classes present in the dataset

    :return: list of class names
    """
    data = read_dataset(features=[])
    return data[LABEL].unique()


def extract_features_and_label(data):
    """Extract features and label from a dataset and return them separately as series

    :param data: dataset as a pd.DataFrame
    :return: features and associated label
    """
    X = data.drop(columns=[LABEL])
    y = data[LABEL]
    return X, y


def extract_scaled_features_and_label(data):
    """Scale and extract features and label from a dataset and return them separately as series

    :param data: dataset as a pd.DataFrame
    :return: scaled features and associated label
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data.drop(LABEL, axis=1))
    scaled_features = scaler.transform(data.drop(LABEL, axis=1))
    df_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])

    X = df_feat
    y = data[LABEL]
    return X, y


def tt_split(data, scaled: bool = False, test_size=0.3, random_state=101):
    if scaled:
        X, y = extract_scaled_features_and_label(data)
    else:
        X, y = extract_features_and_label(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
