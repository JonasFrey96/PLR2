import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import datetime
import sys
import os
import time
import shutil
import argparse
import logging

# misc
import numpy as np
import pandas as pd
import random
import sklearn
from scipy.spatial.transform import Rotation as R
from math import pi
import coloredlogs
import datetime

sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))
# src modules
from helper import pad
from loaders_v2 import ConfigLoader

import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

coloredlogs.install()

from torch.multiprocessing import Process, Queue

# network dense fusion
# from lib.loss import Loss
# from lib.loss_refiner import Loss_refine
from loaders_v2 import GenericDataset
from visu import Visualizer
from helper import re_quat, flatten_dict
from deep_im import DeepIM, ViewpointManager
from helper import BoundingBox
from helper import get_delta_t_in_euclidean
from deep_im import LossAddS
from rotations import *

# move this to seperate file
import matplotlib.pyplot as plt

import signal
import h5py
import glob
from PIL import Image
exp_cfg_path = '/home/jonfrey/PLR2/yaml/exp/exp_ws_deepim.yml'
env_cfg_path = '/home/jonfrey/PLR2/yaml/env/env_natrix_jonas.yml'
exp = ConfigLoader().from_file(exp_cfg_path).get_FullLoader()
env = ConfigLoader().from_file(env_cfg_path).get_FullLoader()


dataset_val = GenericDataset(
    cfg_d=exp['d_train'],
    cfg_env=env)
store = env['p_ycb'] + '/viewpoints_renderings'

# np.array(
#                     self.images[f'{self.store}/{obj}/{best_match}-color.png'])

store = '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/viewpoints_renderings'
ls = glob.glob(f'{store}/*/*.png')

print('Loading all rendered images. This might take a minute')

with h5py.File("/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/test.hdf5", "w") as f_hdf5:
    groups = {}
    for i, f in enumerate(ls):
        if i % 5000 == 0:

            print(f'Loaded {i}/{len(ls)} images')

        k = f.split('/')[-2]
        index = f.split('/')[-1]
        if not k in groups.keys():
            # print('add key', k)
            groups[k] = f_hdf5.create_group(k)
        # print('index', index)
        arr = np.array(Image.open(f))
        img.save(output, format='JPEG')

        groups[k].create_dataset(index, data=arr, dtype=arr.dtype)
