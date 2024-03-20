# imports
import requests
import os
from usleep_api import USleepAPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yasa

stages = ["WAKE", "REM", "N1", "N2", "N3"]

def confidences_to_df(hypnogram):
    return pd.DataFrame(hypnogram, columns=stages)

def stackplot_confidences_from_df(hypno_with_conf, ax, epoch_dur_in_secs=30, xlabel=True):
    epochs_in_hours = [i*epoch_dur_in_secs/60/60 for i in range(len(hypno_with_conf))]
    ax.stackplot(
        epochs_in_hours,                      # x axis
        np.transpose(hypno_with_conf/4*100),  # y axis = data, some math to have it in percent
        labels=stages,
        colors=[[0.6,0.6,0],                  # gold for WAKE
                [1,0.2,0.2],                  # red for REM
                [0.2,0.2,0.8], [0.1,0.2,0.45], [0,0,0.15]]
                                              # shades of blue for NREM
        )
    ax.set_xlim([0, epochs_in_hours[-1]])
    if xlabel: ax.set_xlabel("hours")
    ax.set_ylim([0, 100])
    ax.set_ylabel("% confidence")
    ax.legend()

def hypnogramplot(hypno_no_conf, ax):
    stage_labels = {0: "W", 1: "REM", 2: "N1", 3: "N2", 4: "N3"}
    stages = [stage_labels[e] for e in hypno_no_conf]
    yasa.plot_hypnogram(yasa.Hypnogram(stages), ax=ax)

def stackplot_and_hypnogram(hypno_with_conf, hypno_no_conf):
    f, axs = plt.subplots(2, 1, figsize=(12,4.5), sharex=True)
    stackplot_confidences_from_df(hypno_with_conf, axs[0], xlabel=False)
    hypnogramplot(hypno_no_conf, axs[1])
    plt.tight_layout()
    plt.show()