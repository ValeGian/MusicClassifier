import warnings

import visualize as vl
import features as ft
import classification as cl
import statistics as st
import dataset as dt

import librosa
import numpy as np
import os

DEMONS = './demons_30.wav'

if __name__ == '__main__':
    accuracies = []
    test_num = 10
    labels = dt.get_labels()
    data = dt.read_dataset(labels=['blues', 'classical', 'jazz', 'metal', 'pop', 'rock'])

    '''for i in range(test_num):
        acc = cl.dec_tree(data)[1]
        accuracies.append(acc)
        print(f'{i}: {acc}')
    accuracies = np.array(accuracies)

    avg = np.mean(accuracies)
    print(f'AVG: {avg}')

    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(conf_int)'''

    warnings.filterwarnings("ignore")
    opt_k = 1
    max_acc = 0
    for i in range(1, 10):
        k, acc = cl.elbow_method(data, max_k=20)
        print(f'{k} Neighbors -> {acc}')
        print()
        if acc > max_acc:
            opt_k = k
            max_acc = acc

    print(f'>>>{opt_k} -> {max_acc}')
