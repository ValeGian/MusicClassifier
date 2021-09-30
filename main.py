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
    data = dt.read_dataset()
    for i in range(test_num):
        acc = cl.dec_tree(data)
        accuracies.append(acc)
        print(f'{i}: {acc}')
    accuracies = np.array(accuracies)

    avg = np.mean(accuracies)
    print(f'AVG: {avg}')

    conf_int = st.conf_interval(accuracies, conf_level=0.95)
    print(conf_int)

