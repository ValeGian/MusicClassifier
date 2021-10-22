# MusicClassifier
## Table of Contents
[Overview](#overview)<br/>
[Data Retrieval](#data-retrieval)<br/>
[Feature Extraction](#feature-extraction)<br/>
[Exploratory Data Analysis](#exploratory-data-analysis)<br/>
[Machine Learning](#machine-learning)<br/>
[Summary](#summary)<br/>
[Future Questions](#future-questions)<br/>
[References](#references)<br/>

## Overview
Music is everywhere. As of 2011, it was reported that there are over 79,000,000 documented songs in existence<sup>[1](http://bedtimemath.org/fun-math-songs-in-world/)</sup>. These songs have been classified into hundreds of different genres and subgenres.

[<img src='imgs/music_genre_visualization.png'>](https://musicmachinery.com/2013/09/22/5025/)

And yet, music is still constantly evolving. Some new genres that are expected to see major growth in 2021 are Synthwave, Ambient music, J-pop, and Nu Disco<sup>[3](https://www.ujam.com/blog/upcoming-music-trends-in-2021/)</sup>. With this constant growth and expansion, it's important for businesses that work with music to be able to reliably track these updates.

This project aims to generate accurate and reproducible machine learning models that can predict a song's genre based on its audio features.

## Data Retrieval
In order to conduct this project, I required music data that contains genre labels. The solution I found was through the GTZAN<sup>[4](http://marsyas.info/downloads/datasets.html)</sup> dataset. It consists of 1000 audio tracks, each 30 seconds long. It contains ten genres: blues, classical, country, disco, hip-hop, jazz, reggae, rock, metal, and pop. Each genre consists of 100 sound clips.

## Feature Extraction
Before training the classification model, we have to transform raw data from audio samples into more meaningful representations. Every audio signal consists of many features. However, we must extract the characteristics that are relevant to the problem we are trying to solve.

For extracting such audio features, I decided to use Librosa<sup>[5](https://librosa.org/doc/latest/index.html)</sup> Python module.

The final dataset contains the mean value of each of the following features I chose:
Chroma Frequencies, Root Mean Square, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate, 20 Mel-Frequency Cepstral Coefficients

Once the features have been extracted, we can use existing classification algorithms to classify the songs into different genres, to see which one works best.
.
## Exploratory Data Analysis

## Summary

The goal of this project is to predict the genre of a song based on its audio features. In a machine learning context, this is a multiclass classification problem.

The final dataset that was used to train and test the machine learning models in this project consisted of 1000 songs spanning 10 different genres.

The models I tested in this project were kNN, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and 3 different versions of Support Vector Machines. The supervised machine learning model that performed the best was the Nu-Support Vector classifier<sup>[6](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)</sup> with an accuracy of 68.24% on the whole test set, reaching 89.12% on a specific subset of it.

## Future Ideas

This project gave a good baseline for genre classification on single label classification. While this is useful, we know that songs can have multiple genres. Songs are not discreet and do not strictly fall into one genre or another. Looking forward, I would like to try to extend this model to perform multi-genre (multi-label) classification.

Another avenue I'm interested in exploring is genre classification directly from raw audio data through Deep Learning techniques. It would be interesting to see if there's enough inherent information in song lyrics to build NLP models that can outperform audio feature models.

If you made it this far, thanks so much for reading!

## References

1. [79,000,000 Song Metric](http://bedtimemath.org/fun-math-songs-in-world/)
2. [Music Genre Visualization](https://musicmachinery.com/2013/09/22/5025/)
3. [2021 Music Genre Trends](https://www.ujam.com/blog/upcoming-music-trends-in-2021/)
4. [GTZAN dataset](http://marsyas.info/downloads/datasets.html)
5. [Librosa Python module](https://librosa.org/doc/latest/index.html)
6. [Nu-Support Vector classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)