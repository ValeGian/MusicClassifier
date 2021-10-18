import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import sklearn
import seaborn as sns

def correlation_matrix(data):
    sns.set_theme(style="white")
    # Compute the correlation matrix
    selected_cols = data.columns.tolist()
    corr = data[selected_cols].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(19, 14), dpi=200)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 15, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})

    plt.savefig('./imgs/audio_features_correlation_matrix.png')

# Helper to plot confusion matrix -- from Scikit-learn website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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
