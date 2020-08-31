import numpy as np
import os
import sys
sys.path.append('/home/jonfrey/PLR2/src/')
sys.path.append('/home/jonfrey/PLR2/src/dense_fusion')
sys.path.append('/home/jonfrey/PLR2/src/visu')
from math import pi
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import re_quat, rotation_angle
from estimation.filter import Linear_Estimator, Kalman_Filter
from estimation.state import State_R3xQuat, State_SE3, points
from estimation.errors import ADD, ADDS, translation_error, rotation_error
from visu import plot_pcd, SequenceVisualizer
from copy import deepcopy
import pandas as pd
import copy
import glob

sym_list = [12, 15, 18, 19, 20]


def extract_data(seq_data, name):
    data_agg = []
    for i in range(len(seq_data)):
        seq = seq_data[i]
        for t in range(len(seq)):
            frame = seq[t]
            idx = frame['dl_dict']['idx'][0][0]
            if idx in sym_list:
                symmetric = True
            else:
                symmetric = False
            entry = (i, t, idx, symmetric,
                     frame['ADD'], frame['t_error'], frame['rot_error'])
            data_agg.append(entry)

    df = pd.DataFrame(data_agg, columns=[
                      'seq', 'frame', 'idx', 'symmetric', 'ADD', 't_error', 'rot_error'])
    df['model'] = name
    return df


def measure_compare_models_objects(df):
    df_group = df.groupby(['model', 'idx'])
    df_new = df_group.agg(['mean'])
    df_new.drop(['seq', 'frame'], axis=1, inplace=True)
    return df_new


def measure_compare_models(df):
    df_group = df.groupby(['model'])
    df_new = df_group.agg(['mean'])
    df_new.drop(['seq', 'frame'], axis=1, inplace=True)
    return df_new


def metrics_by_object(df):
    df_group = df.groupby(['model', 'idx'])
    df_new = df_group.agg(['mean', 'count'])
    df_new.drop(['seq', 'frame'], axis=1, inplace=True)
    return df_new


def metrics_symmetric(df):
    df_new = df.groupby('symmetric').mean()
    df_new.drop(['seq', 'frame', 'idx'], axis=1, inplace=True)
    return df_new


def metrics_by_sequence(df):
    df_new = df.groupby('seq').mean()
    df_new.drop(['frame'], axis=1, inplace=True)
    return df_new


def plot_stacked_histogram(df, column, measure, legend=True):
    fig, ax = plt.subplots(figsize=[8, 6])
    data_list = []
    legend_list = []
    for c in df[column].unique():
        indexes = df[column] == c
        data_list.append(df[indexes][measure])
        legend_list.append(c)

    n, bins, patches = ax.hist(
        data_list, histtype='barstacked', linewidth=0.2, edgecolor='black', density=True)
    ax.set_xlabel(measure)
    ax.set_ylabel('Frequency')
    if legend:
        ax.legend(legend_list)
    return fig, ax


def plot_histogram(df, measure, legend=True):
    fig, ax = plt.subplots(figsize=[8, 6])
    data = df[measure]

    n, bins, patches = ax.hist(
        data, histtype='bar', linewidth=0.2, edgecolor='black', density=True)
    ax.set_xlabel(measure)
    ax.set_ylabel('Frequency')
    return fig, ax
