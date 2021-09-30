import features as ft

import os
import csv
import librosa
import pandas as pd

DATASET_PATH = './dataset'
DATASET = 'data.csv'
LABEL = 'label'


def write_dataset(header: list[str]):
    '''Extract features from the audio signals in the original dataset

    In output we will have the data.csv file, that will be our dataset

    :param header: list of feature names
    :return:

    Examples
    --------
    >>> import features as ft
    >>> features = [ft.Features.CHROMAGRAM, ft.Features.MFCC]
    >>> n_mfcc = 15
    >>> exp_features = ft.explode_features(features, n_mfcc=n_mfcc)
    >>> create_features_dataset(exp_features)
    '''

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
    '''Read specified features from dataset

    :param features: list of features to read
    :param labels: list of class names. Used to select specific rows. [] = select every class
    :return: dataset as a pd.DataFrame
    '''
    names = ft.explode_features(ft.DEF_FEATURES, n_mfcc=ft.DEF_N_MFCC)
    selected_cols = ft.explode_features(features, n_mfcc=n_mfcc)
    data = pd.read_csv(DATASET, header=None, skipinitialspace=True, names=names, usecols=selected_cols)
    if labels:
        data = data[data[LABEL].isin(labels)]
    else:
        data = data.drop(labels=0, axis=0)  # drop row 0 of the dataframe
    return data

def get_labels():
    '''Get label classes present in the dataset

    :return: list of class names
    '''
    data = read_dataset(features=[])
    return data[LABEL].unique()
