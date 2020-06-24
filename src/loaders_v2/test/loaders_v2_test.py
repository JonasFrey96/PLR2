import os
import sys
import numpy as np

if __name__ == "__main__":
    os.chdir('/home/jonfrey/PLR')
    sys.path.append('src')
    sys.path.append('src/dense_fusion')

from loaders_v2 import Backend, ConfigLoader, GenericDataset
from visu import Visualizer
from PIL import Image
import copy
from helper import re_quat
from scipy.spatial.transform import Rotation as R

import argparse

def load_flags():
    parser = argparse.ArgumentParse()
    parser.add_argument('--env', type=str, help='Environment config file. Same as for tools/lightning.py',
            default=os.environ['TRACK_ENV_CONFIG_FILE'])
    parser.add_argument('--dataset', type=str, help='Dataset config file.',
            default="src/loaders_v2/test/dataset_cfgs.yaml")
    return parser.parse_args()

if __name__ == "__main__":
    flags = load_flags()
    dataset_configuration = ConfigLoader().from_file(flags.dataset)
    environment_configuration = ConfigLoader().from_file(flags.env)

    generic = GenericDataset(
        cfg_d=dataset_configuration['d_ycb'],
        cfg_env=environment_configuration)

    # ycb = YCBDataset(cfg_d=exp_cfg['d_YCB'],
    #                  cfg_env=env_cfg, refine=False, visu=True)
    # generic.refine = True
    # seq = generic[0]
    # print("Should be refine size:", seq[0][3].shape)
    # generic.refine = False
    # seq = generic[0]
    # print("Should be normal size:", seq[0][3].shape)
    # generic.visu = True
    # seq = generic[0]
    # print("Should be an image:", seq[0][8].shape)
    # generic.visu = False
    # seq = generic[0]
    # print("Should be 0 since no image found:", seq[0][8])
    # generic.visu = True
    for i in range(0, 10):
        frame = generic[i][0]

        # frame2 = dataset_train[0][0]
        # frame = ycb[0][0]
        dl_dict = {}
        keys = ['points', 'choose', 'img', 'target', 'model_points', 'idx', 'ff_trans',
                'ff_rot', 'depth_img', 'img_org', 'cam_cal', 'gt_rot_wxyz', 'gt_trans']

        for _j, _i in enumerate(keys):
            dl_dict[_i] = frame[_j]
        p_res = '~/images/'
        os.makedirs(p_res, exist_ok=True)
        visu = Visualizer(p_res)
        img = np.array(dl_dict['img'])
        img = np.transpose(img, (2, 1, 0)).astype(np.uint8)
        Image.fromarray(img).save(f'{p_res}/img_croped.png', "png")

        img_orgi = dl_dict['img_org']
        cam = dl_dict['cam_cal']
        target = np.array(dl_dict['target'])
        points = np.array(dl_dict['points'])
        model_points = np.array(dl_dict['model_points'])

        visu.plot_estimated_pose('Target%s' % i, 1, copy.deepcopy(img_orgi), target, np.zeros((1, 3)), np.eye(3),
                                 cam_cx=cam[0],
                                 cam_cy=cam[1],
                                 cam_fx=cam[2],
                                 cam_fy=cam[3],
                                 store=True,
                                 jupyter=False, w=1)
        visu.plot_estimated_pose('Points%s' % i, 1, copy.deepcopy(img_orgi), points, np.zeros((1, 3)), np.eye(3),
                                 cam_cx=cam[0],
                                 cam_cy=cam[1],
                                 cam_fx=cam[2],
                                 cam_fy=cam[3],
                                 store=True,
                                 jupyter=False, w=1)
        trans = np.array(dl_dict['gt_trans'])
        rot = np.array(dl_dict['gt_rot_wxyz'])
        quat = re_quat(rot, 'wxyz').tolist()
        rot = R.from_quat(quat)
        visu.plot_estimated_pose('ModelPoints%s' % i, 1, copy.deepcopy(img_orgi), model_points, trans.reshape(1, 3), rot.as_matrix(),
                                 cam_cx=cam[0],
                                 cam_cy=cam[1],
                                 cam_fx=cam[2],
                                 cam_fy=cam[3],
                                 store=True,
                                 jupyter=False, w=1)
