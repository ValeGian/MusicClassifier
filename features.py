from enum import Enum
import librosa
import numpy as np


class Features(Enum):
    CHROMAGRAM = 'chroma_stft',
    RMS = 'rms',
    SPEC_CENT = 'spectral_centroid',
    SPEC_BW = 'spectral_bandwidth',
    ROLLOFF = 'rolloff',
    ZCR = 'zero_crossing_rate',
    MFCC = 'mfcc'


DEF_FEATURES = [Features.CHROMAGRAM, Features.RMS, Features.SPEC_CENT, Features.SPEC_BW, Features.ROLLOFF, Features.ZCR,
                Features.MFCC]
DEF_N_MFCC = 20


def extract_features(x: any, sr: any = 22050, features: list[Features] = DEF_FEATURES, n_mfcc: int = DEF_N_MFCC):
    # x is the floating point time series of the audio
    # sr is the sampling rate
    extracted = []

    if Features.CHROMAGRAM in features:
        chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
        extracted.append(np.mean(chroma_stft))

    if Features.RMS in features:
        rms = librosa.feature.rms(x)
        extracted.append(np.mean(rms))

    if Features.SPEC_CENT in features:
        spec_cent = librosa.feature.spectral_centroid(x, sr=sr)
        extracted.append(np.mean(spec_cent))

    if Features.SPEC_BW in features:
        spec_bw = librosa.feature.spectral_bandwidth(x, sr=sr)
        extracted.append(np.mean(spec_bw))

    if Features.ROLLOFF in features:
        rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
        extracted.append(np.mean(rolloff))

    if Features.ZCR in features:
        zcr = librosa.feature.zero_crossing_rate(x)
        extracted.append(np.mean(zcr))

    if Features.MFCC in features:
        mfcc = librosa.feature.mfcc(x, sr=sr, n_mfcc=n_mfcc)
        for e in mfcc:
            extracted.append(np.mean(e))

    return extracted


def explode_features(features: list[Features], n_mfcc: int = DEF_N_MFCC) -> list[str]:
    exploded_features = []
    for feature in features:
        if feature != Features.MFCC:
            exploded_features.append(feature.name.lower())
        else:
            for i in range(n_mfcc):
                exploded_features.append(f'{Features.MFCC.name.lower()}{i}')
    exploded_features.append('label')
    return exploded_features
