import os
import numpy as np
import torch
import sys
import argparse
import copy
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))
sys.path.append(os.path.join(os.getcwd() + '/src/deep_im/lib'))
from loaders_v2 import ConfigLoader
from loaders_v2 import GenericDataset
from src.deep_im.lib.render_glumpy.render_py_light import Render_Py_Light


def mat2quat(M):
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = (
        np.array(
            [
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz],
            ]
        )
        / 3.0
    )
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    if q[0] < 0:
        q *= -1
    return q


class RendererYCB():
    def __init__(self, p_ycb, obj_name_2_idx, K1, K2):
        # loads all models
        self.renderes = {}
        K = np.array([[1066.778, 0, 312.9869], [
                     0, 1067.487, 241.3109], [0, 0, 1]])
        ZNEAR = 0.25
        ZFAR = 6.0
        for name, idx in obj_name_2_idx.items():
            model_dir = f'{p_ycb}/models/{name}'
            width = 640
            height = 480
            brightness_ratios = [0.3]
            # add for each camera calibration a costum renderer
            self.renderes[idx] = [Render_Py_Light(model_dir, K1, width, height, ZNEAR, ZFAR, brightness_ratios=brightness_ratios),
                                  Render_Py_Light(model_dir, K2, width, height, ZNEAR, ZFAR, brightness_ratios=brightness_ratios)]

    def render(self, obj_idx, r_mat, trans, noise, cam):
        """[summary]

        Parameters
        ----------
        obj_idx : int
            0 - (max_obj-1)
        r_mat : np.array 3x3
            [description]
        trans : np.array 3
            translaton xyz
        noise : [type]
            [description]
        cam : int
            0 - 1 what set of camera parameters should be used
        """

        rend = self.renderes[obj_idx][cam]
        r_quat = mat2quat(r_mat)
        rgb, depth = rend.render(r_quat, t, light_position=[
            0, 0, -1], light_intensity=[0, 0, 0], brightness_k=0)
        # how can i verfiy that anything works ?


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws_motion_train.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_flags()
    exp = ConfigLoader().from_file(args.exp).get_FullLoader()
    env = ConfigLoader().from_file(args.env).get_FullLoader()
    dataset_train = GenericDataset(
        cfg_d=exp['d_train'],
        cfg_env=env)
    K1 = np.array([[1066.778, 0, 312.9869], [
        0, 1067.487, 241.3109], [0, 0, 1]])
    K2 = np.array([[1066.778, 0, 312.9869], [
        0, 1067.487, 241.3109], [0, 0, 1]])

    obj_name_2_idx = copy.deepcopy(dataset_train._backend._name_to_idx)

    RendererYCB('/media/scratch1/jonfrey/datasets/YCB_Video_Dataset',
                obj_name_2_idx=obj_name_2_idx,
                K1=K1,
                K2=K2)
