import numpy as np
import os
import sys
sys.path.append('/home/jonfrey/PLR/src/')
sys.path.append('/home/jonfrey/PLR/src/dense_fusion')
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
from scipy.spatial.transform import Rotation as R
import pandas as pd
import pickle as pkl
import copy
import glob
import k3d

sym_list = [12, 15, 18, 19, 20]


def kf_sequence(data_old, var_motion, var_sensor, params):
    # extract relevant data

    data = copy.deepcopy(data_old)
    data = list(data)
    for i, seq in enumerate(data):
        # print('Loading sequence {}'.format(i))
        # setup of filter
        idx = np.squeeze(seq[0]['dl_dict']['idx'])
        model_points_np = np.squeeze(seq[0]['dl_dict']['model_points'])
        model_points = points(model_points_np)

        # set up kalman filter
        prior = State_R3xQuat([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        prior.add_noise([0, 0, 0, 0, 2 * pi, 3])
        prior_variance = State_R3xQuat([10, 10, 10, 10, 10, 10, 20 * pi])
        prior_variance = prior_variance.state.reshape(7,)
        kf = Kalman_Filter(state_type='State_R3xQuat',
                           motion_model='Linear_Motion',
                           trajectory=None,
                           observations=None,
                           variance_motion=var_motion,
                           variance_sensor=var_sensor,
                           params=params)
        kf.set_prior(prior, prior_variance)

        for obs_data in seq:
            # run the kalman filter over the observation
            obs_t = np.array(obs_data['final_pred_obs']['t'])
            obs_r = re_quat(
                np.array(obs_data['final_pred_obs']['r_wxyz']), 'wxyz')
            obs_c = obs_data['final_pred_obs']['c']
            gt_pose = State_R3xQuat(np.concatenate((obs_data['dl_dict']['gt_trans'][0],
                                                    re_quat(copy.deepcopy(obs_data['dl_dict']['gt_rot_wxyz'][0]), 'wxyz')), axis=0))
            # need to convert quat to xyzw
            obs = State_R3xQuat(np.concatenate((obs_t, obs_r), axis=0))
            variance = np.true_divide(var_sensor, obs_c)
            f_pose, _ = kf.update(obs, variance)
            kf.predict()
            filter_pred = {'t': f_pose.state[0:3, 0], 'r_wxyz': re_quat(
                copy.deepcopy(f_pose.state[3:7, 0]), 'xyzw')}

            # calculate the ADD error
            if idx in sym_list:
                filter_error_ADD = ADDS(model_points, gt_pose, f_pose)
            else:
                filter_error_ADD = ADD(model_points, gt_pose, f_pose)
            obs_data['filter_pred'] = filter_pred
            obs_data['ADD'] = filter_error_ADD
            obs_data['translation_error'] = translation_error(gt_pose, f_pose)
            obs_data['rotation_error'] = rotation_error(gt_pose, f_pose)
        # print('Processed sequence.')

    return data
