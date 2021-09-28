from enum import Enum
import librosa
import numpy as np

import pandas as pd
import os
import csv

class Features(Enum):
    CHROMAGRAM = 'chroma_stft',
    RMS = 'rms',
    SPEC_CENT = 'spectral_centroid',
    SPEC_BW = 'spectral_bandwidth',
    ROLLOFF = 'rolloff',
    ZCR = 'zero_crossing_rate',
    MFCC = 'mfcc'

FEATURES_TO_CONSIDER = [Features.CHROMAGRAM, Features.RMS, Features.SPEC_CENT, Features.SPEC_BW, Features.ROLLOFF, Features.ZCR, Features.MFCC]
DEF_N_MFCC = 20

DATASET_PATH = './dataset'

def extract_features(x: any, sr: any = 22050, n_mfcc: int = DEF_N_MFCC):
    # x is the floating point time series of the audio
    # sr is the sampling rate
    extracted = []

    if Features.CHROMAGRAM in FEATURES_TO_CONSIDER:
        chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
        extracted.append(np.mean(chroma_stft))

    if Features.RMS in FEATURES_TO_CONSIDER:
        rms = librosa.feature.rms(x)
        extracted.append(np.mean(rms))

    if Features.SPEC_CENT in FEATURES_TO_CONSIDER:
        spec_cent = librosa.feature.spectral_centroid(x, sr=sr)
        extracted.append(np.mean(spec_cent))

    if Features.SPEC_BW in FEATURES_TO_CONSIDER:
        spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
        extracted.append(np.mean(spec_bw))

    if Features.ROLLOFF in FEATURES_TO_CONSIDER:
        rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
        extracted.append(np.mean(rolloff))

    if Features.ZCR in FEATURES_TO_CONSIDER:
        zcr = librosa.feature.zero_crossing_rate(x)
        extracted.append(np.mean(zcr))

    if Features.MFCC in FEATURES_TO_CONSIDER:
        mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc)
        for e in mfcc:
            extracted.append(np.mean(e))

    return extracted

def  create_features_dataset(n_mfcc: int = DEF_N_MFCC):
    """
    Function used to extract all the possible features from the audio signals in the original dataset
    In output we will have the data.csv file, that will be our dataset
    """

    # construct the header of the table
    header = []
    for feature in FEATURES_TO_CONSIDER:
        if feature != Features.MFCC:
            header.append(feature.name.lower())
        else:
            for i in range(n_mfcc):
                header.append(f'{Features.MFCC.name.lower()}{i}')
    header.append('label')

    # data.csv will be the new dataset
    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # add a row for each song in the original dataset
    count = 0
    for genre in os.listdir(DATASET_PATH):
        print(genre)
        for filename in os.listdir(f'{DATASET_PATH}/{genre}'):
            song_path = f'{DATASET_PATH}/{genre}/{filename}'
            x, sr = librosa.load(song_path)
            features = extract_features(x, sr=sr)
            features.append(genre) # add the label

            # Insert the new calculated values of this song in the file
            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(features)
            count += 1
            print(count)