import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def find_best_pos(data1, data2):
    n1 = len(data1_key)
    n2 = len(data2_key)


if __name__ == "__main__":
    data1_key = pd.read_csv("data_cmd.csv", index_col=0)
    data2_key = pd.read_csv("data_qw.csv", index_col=0)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data1_key.values.reshape(-1), '.-')
    axs[0].set_xticks(np.arange(0, len(data2_key), 24))
    axs[0].set_xticklabels(
        list(map(lambda x: x[5:-3], data1_key.index[::24])), rotation=0)
    axs[0].set_title("data from cma")

    axs[1].plot(data2_key.values.reshape(-1), '.-')
    axs[1].set_xticks(np.arange(0, len(data2_key), 24))
    axs[1].set_xticklabels(
        list(map(lambda x: x[5:-3], data2_key.index[::24])), rotation=0)
    cor = np.corrcoef(data1_key.values.reshape(-1),
                      data2_key.values.reshape(-1))[1, 0]
    axs[1].set_title("data from hefeng, corr coef is %.2f" % cor)
    plt.tight_layout()
    data1_key.to_csv("data_cmd.csv")
    data2_key.to_csv("data_qw.csv")

    plt.show()

    n1 = len(data1_key)
    n2 = len(data2_key)
    cors = []
    for i in range(min(24, n2)):
        cor = np.corrcoef(data1_key.values.reshape(-1)[:n2-i],
                          data2_key.values.reshape(-1)[i:])[1, 0]
        cors.append(cor)
    plt.plot(cors)
    plt.show()
    ind = np.argmax(cors)
    plt.plot(data1_key.values.reshape(-1)[:n2 - ind])
    plt.plot(data2_key.values.reshape(-1)[ind:])
    plt.show()
