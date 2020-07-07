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
print(os.path.join(os.getcwd() + '/src'))
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

# network dense fusion
from lib.loss import KeypointLoss
from lib.loss_refiner import Loss_refine
from lib.network import KeypointNet, PoseRefineNet

# dataset
from loaders_v2 import GenericDataset
from visu import Visualizer
from helper import re_quat, flatten_dict


class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()

        # logging h-params
        exp_config_flatten = flatten_dict(exp.get_FullLoader())
        for k in exp_config_flatten.keys():
            if exp_config_flatten[k] is None:
                exp_config_flatten[k] = 'is None'
        self.hparams = exp_config_flatten

        self.test_size = 0.9
        self.env, self.exp = env, exp

        self.estimator = KeypointNet(
            num_points=exp['d_train']['num_points'],
            num_obj=exp['d_train']['objects'])

        if exp['estimator_restore']:
            try:
                self.estimator.load_state_dict(torch.load(
                    exp['estimator_load']))
            except:
                state_dict = torch.load(exp['estimator_load'])
                self.load_my_state_dict(state_dict)

        num_poi = exp['d_train']['num_pt_mesh_small']
        self.criterion = KeypointLoss(num_poi)
        num_poi = exp['d_train']['num_pt_mesh_large']
        self.criterion_refine = Loss_refine(
            num_poi, exp['d_train']['obj_list_sym'])

        self.refine = False
        self.w = exp['w_normal']

        self.best_validation = 999
        self.best_validation_patience = 5
        self.early_stopping_value = 0.1

        self.Visu = None
        self._dict_track = {}

        self.number_images_log_val = 5
        self.number_images_log_test = 10
        self.counter_images_logged = 0

        self.init_train_vali_split = False

    def load_my_state_dict(self, state_dict):
        own_state = self.estimator.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if param.shape != own_state[name].shape:
                _a, _b, _c = param.shape
                own_state[name][:_a, :_b, :_c] = param
                print(name, ': ', own_state[name].shape,
                      ' mergerd with ', param.shape)
            else:
                own_state[name] = param
                print('worked for ', name)

    def forward(self, img, points, choose, idx):
        return self.estimator(
            img, points, choose, idx)

    def training_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0
        l = len(batch)
        for frame in batch:
            # unpack the batch and apply forward pass
            if frame[0].dtype == torch.bool:
                continue

            (points, choose, img, gt_keypoints, model_points, idx,
                    depth_img, img_orig, cam,
                    gt_rot_wxyz, gt_trans, unique_desig) = frame

            predicted_keypoints, emb = self(img, points, choose, idx)

            loss, dis = self.criterion(
                predicted_keypoints, gt_keypoints, points, gt_trans)
            total_loss += loss
            total_dis += dis
        # choose correct loss here
        total_loss = total_loss / l
        total_dis = total_dis / l
        tensorboard_logs = {'train_loss': total_loss, 'train_dis': total_dis}
        return {'loss': total_loss, 'dis': total_dis, 'log': tensorboard_logs, 'progress_bar': {'train_dis': total_dis, 'train_loss': total_loss}}

    def validation_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0

        for frame in batch:

            if frame[0].dtype == torch.bool:
                continue

            # unpack the batch and apply forward pass
            (points, choose, img, gt_keypoints, model_points, idx,
                    depth_img, img_orig, cam,
                    gt_rot_wxyz, gt_trans, unique_desig) = frame

            predicted_keypoints, emb = self(img, points, choose, idx)

            loss, dis = self.criterion(
                predicted_keypoints, gt_keypoints, points, gt_trans)

            if f'val_loss' in self._dict_track.keys():
                self._dict_track[f'val_loss'].append(
                    float(loss))
                self._dict_track[f'val_dis'].append(
                    float(dis))
            else:
                self._dict_track[f'val_loss'] = [float(loss)]
                self._dict_track[f'val_dis'] = [float(dis)]

            if f'val_{int(unique_desig[1])}_loss' in self._dict_track.keys():
                self._dict_track[f'val_{int(unique_desig[1])}_loss'].append(
                    float(loss))
                self._dict_track[f'val_{int(unique_desig[1])}_dis'].append(
                    float(dis))
            else:
                self._dict_track[f'val_{int(unique_desig[1])}_loss'] = [
                    float(loss)]
                self._dict_track[f'val_{int(unique_desig[1])}_dis'] = [
                    float(dis)]

            if self.number_images_log_val > self.counter_images_logged:
                # self.visu(batch_idx, pred_r, pred_t, pred_c, points,
                #           target, model_points, cam, img_orig, unique_desig)
                self.counter_images_logged += 1

            total_loss += loss
            total_dis += dis

        tensorboard_logs = {'val_loss': total_loss /
                            len(batch), 'val_dis': total_dis / len(batch)}
        return {'val_loss': total_loss / len(batch), 'val_dis': total_dis / len(batch), 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # used for tensorboard logging
        total_loss = 0
        total_dis = 0

        for frame in batch:

            if frame[0].dtype == torch.bool:
                continue

            # unpack the batch and apply forward pass
            (points, choose, img, gt_keypoints, model_points, idx,
                    depth_img, img_orig, cam,
                    gt_rot_wxyz, gt_trans, unique_desig) = frame

            predicted_keypoints, emb = self(img, points, choose, idx)

            loss, dis = self.criterion(
                predicted_keypoints, gt_keypoints, points, gt_trans)  # wxy

            # self._dict_track is used to log hole epoch
            # add dis and loss to dict
            if f'test_loss' in self._dict_track.keys():
                self._dict_track[f'test_loss'].append(
                    float(loss))
                self._dict_track[f'test_dis'].append(
                    float(dis))
            else:
                self._dict_track[f'test_loss'] = [float(loss)]
                self._dict_track[f'test_dis'] = [float(dis)]

            # add all object losses individual to dataframe
            if f'test_{int(unique_desig[1])}_loss' in self._dict_track.keys():
                self._dict_track[f'test_{int(unique_desig[1])}_loss'].append(
                    float(loss))
                self._dict_track[f'test_{int(unique_desig[1])}_dis'].append(
                    float(dis))
            else:
                self._dict_track[f'test_{int(unique_desig[1])}_loss'] = [
                    float(loss)]
                self._dict_track[f'test_{int(unique_desig[1])}_dis'] = [
                    float(dis)]

            if self.number_images_log_test > self.counter_images_logged:
                # self.visu(batch_idx, pred_r, pred_t, pred_c, points,
                #           target, model_points, cam, img_orig, unique_desig)
                self.counter_images_logged += 1

            total_loss += loss
            total_dis += dis

        tensorboard_logs = {'test_loss': total_loss / len(batch),
                            'test_dis': total_dis / len(batch)}

        return {**tensorboard_logs,
                'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_dict = {}
        for old_key in list(self._dict_track.keys()):
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
        self._dict_track = {}

        return {
            **avg_dict, 'log': avg_dict}

    def validation_epoch_end(self, outputs):
        avg_dict = {}
        for old_key in list(self._dict_track.keys()):
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
        self._dict_track = {}

        if avg_dict['avg_val_dis'] < self.best_validation:

            self.best_validation = avg_dict['avg_val_dis']
            self.best_validation_patience_run = 0
        else:
            self.best_validation_patience_run += 1
            if self.best_validation_patience_run > self.best_validation_patience:
                print("figure out how to set stop training flag")

        if avg_dict['avg_val_dis'] < self.early_stopping_value:
            print("figure out how to set stop training flag")

        self.counter_images_logged = 0  # reset image log counter

        tensorboard_log = copy.deepcopy(avg_dict)
        for k in avg_dict.keys():
            avg_dict[k] = torch.tensor(
                avg_dict[k], dtype=torch.float32, device=self.device)

        return {**avg_dict, 'avg_val_dis_float': float(avg_dict['avg_val_dis']), 'log': tensorboard_log}

    def visu(self, batch_idx, pred_r, pred_t, pred_c, points, target, model_points, cam, img_orig, unique_desig):
        if self.Visu is None:
            self.Visu = Visualizer(exp['model_path'] +
                                   '/visu/', self.logger.experiment)
        self.Visu.plot_estimated_pose(tag='ground_truth_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                      epoch=self.current_epoch,
                                      img=img_orig[0, :, :,
                                                   :].detach().cpu().numpy(),
                                      points=copy.deepcopy(
            target[0, :, :].detach().cpu().numpy()),
            trans=np.array([[0, 0, 0]]),
            rot_mat=np.diag((1, 1, 1)),
            cam_cx=float(cam[0, 0]),
            cam_cy=float(cam[0, 1]),
            cam_fx=float(cam[0, 2]),
            cam_fy=float(cam[0, 3]),
            store=True)
        # extract highest confident vote
        how_max, which_max = torch.max(pred_c, 1)
        div = (torch.norm(pred_r, dim=2).view(1, 1000, 1))
        pred_r = (-1) * pred_r / div

        c = how_max.detach()
        t = (pred_t[0, int(which_max), :] +
             points[0, int(which_max), :]).detach().cpu().numpy()
        r = pred_r[0, int(which_max), :].detach().cpu().numpy()

        rot = R.from_quat(re_quat(r, 'wxyz'))

        self.Visu.plot_estimated_pose(tag='final_prediction_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                      epoch=self.current_epoch,
                                      img=img_orig[0, :, :,
                                                   :].detach().cpu().numpy(),
                                      points=copy.deepcopy(
                                          model_points[0, :, :].detach(
                                          ).cpu().numpy()),
                                      trans=t.reshape((1, 3)),
                                      rot_mat=rot.as_matrix(),
                                      cam_cx=float(cam[0, 0]),
                                      cam_cy=float(cam[0, 1]),
                                      cam_fx=float(cam[0, 2]),
                                      cam_fy=float(cam[0, 3]),
                                      store=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.estimator.parameters(), lr=self.exp['lr'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.exp['lr_cfg']['on_plateau_cfg']),
            'monitor': 'avg_val_dis',  # Default: val_loss
            'interval': self.exp['lr_cfg']['interval'],
            'frequency': self.exp['lr_cfg']['frequency']
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset_train = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        # initalize train and validation indices
        if not self.init_train_vali_split:
            self.init_train_vali_split = True
            self.indices_valid, self.indices_train = sklearn.model_selection.train_test_split(
                range(0, len(dataset_train)), test_size=self.test_size)

        dataset_subset = torch.utils.data.Subset(
            dataset_train, self.indices_train)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=self.exp['loader']['batch_size'],
                                                       shuffle=True,
                                                       num_workers=self.exp['loader']['workers'],
                                                       pin_memory=True)
        return dataloader_train

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_test'],
            cfg_env=self.env)

        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=self.exp['loader']['batch_size'],
                                                      shuffle=False,
                                                      num_workers=self.exp['loader']['workers'],
                                                      pin_memory=True)
        return dataloader_test

    def val_dataloader(self):
        dataset_val = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        # initalize train and validation indices
        if not self.init_train_vali_split:
            self.init_train_vali_split = True
            self.indices_valid, self.indices_train = sklearn.model_selection.train_test_split(
                range(0, len(dataset_val)), test_size=self.test_size)

        dataset_subset = torch.utils.data.Subset(
            dataset_val, self.indices_valid)
        dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                     batch_size=self.exp['loader']['batch_size'],
                                                     shuffle=False,
                                                     num_workers=self.exp['loader']['workers'],
                                                     pin_memory=True)
        return dataloader_val


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    parser.add_argument('-w', '--workers', default=None)
    return parser.parse_args()

if __name__ == "__main__":
    # for reproducability
    seed_everything(42)

    args = read_args()
    exp_cfg_path = args.exp
    env_cfg_path = args.env

    exp = ConfigLoader().from_file(exp_cfg_path)
    env = ConfigLoader().from_file(env_cfg_path)

    if args.workers is not None:
        # This is for debugging. Can easily set workers to 0 so data is loaded on the main thread and
        # the debugger can be loaded.
        exp['loader']['workers'] = int(args.workers)

    """
    Trainer args (gpus, num_nodes, etc…) && Program arguments (data_path, cluster_email, etc…)
    Model specific arguments (layer_dim, num_layers, learning_rate, etc…)
    """
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()

    p = exp['model_path'].split('/')
    p.append(str(timestamp) + '_' + p.pop())
    new_path = '/'.join(p)
    exp['model_path'] = new_path
    model_path = exp['model_path']

    logger = logging.getLogger('TrackNet')

    # copy config files to model path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        logger.info((pad("Generating network run folder")))
    else:
        logger.warning((pad("Network run folder already exits")))

    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]

    logger.info(pad(f'Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}'))
    shutil.copy(exp_cfg_path, f'{model_path}/{exp_cfg_fn}')
    shutil.copy(env_cfg_path, f'{model_path}/{env_cfg_fn}')

    dic = {'exp': exp, 'env': env}
    model = TrackNet6D(**dic)

    # default used by the Trainer
    # TODO create one early stopping callback
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/63bd0582e35ad865c1f07f61975456f65de0f41f/pytorch_lightning/callbacks/base.py#L12
    early_stop_callback = EarlyStopping(
        monitor='avg_val_dis_float',
        patience=exp.get('early_stopping_cfg', {}).get('patience', 10),
        strict=True,
        verbose=True,
        mode='min'
    )

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        verbose=True,
        monitor="avg_val_dis",
        mode="min",
        prefix="",
    )

    trainer = Trainer(gpus=1,
                      num_nodes=1,
                      auto_lr_find=False,
                      accumulate_grad_batches=exp['accumulate_grad_batches'],
                      default_root_dir=model_path,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback,
                      fast_dev_run=True,
                      limit_train_batches=1.0,
                      limit_test_batches=1.0,
                      limit_val_batches=1.0,
                      val_check_interval=0.0,
                      terminate_on_nan=True)

    trainer.fit(model)
    trainer.test(model)
