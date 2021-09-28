import matplotlib.pyplot as plt
import librosa.display
import sklearn

def waveform(x: any, sr: any = 22050):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()

def spectogram(x: any, sr: any = 22050):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def zero_crossing_rate(x: any):
    zero_crossings = librosa.zero_crossings(x[0:x.size], pad=False)
    print(f'Zero crossings: ' + f'{sum(zero_crossings)}')

def spectral_centroids(x: any, sr: any = 22050):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    # Plotting the Spectral Centroid along the waveform
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    plt.plot(t, normalize(spectral_centroids), color='r')
    plt.show()

def spectral_rolloff(x: any, sr: any = 22050):
    spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
    librosa.display.waveplot(x, sr=sr, alpha=0.4)
    # Computing the time variable for visualization
    frames = range(len(spectral_rolloff))
    t = librosa.frames_to_time(frames)
    plt.plot(t, normalize(spectral_rolloff), color='r')
    plt.show()

def mfccs(x: any, sr: any = 22050):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    # Perform feature scaling such that each coefficient dimension has zero mean and unit variance
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()

def chromagram(x: any, sr: any = 22050):
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    plt.show()

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
