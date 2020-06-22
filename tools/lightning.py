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
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

coloredlogs.install()

# network dense fusion
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.network import PoseNet, PoseRefineNet

# dataset
from loaders_v2 import GenericDataset
from visu import Visualizer
from helper import re_quat


class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self.hparams = {'exp': exp, 'env': env}

        self.env = env
        self.exp = exp

        self.estimator = PoseNet(
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
        self.criterion = Loss(num_poi, exp['d_train']['obj_list_sym'])
        num_poi = exp['d_train']['num_pt_mesh_large']
        self.criterion_refine = Loss_refine(
            num_poi, exp['d_train']['obj_list_sym'])

        self.refine = False
        self.w = exp['w_normal']

        self.best_validation = 999
        self.best_validation_patience = 5
        self.early_stopping_value = 0.1
        self.Visu = Visualizer(exp['model_path'] + '/visu/')
        self._dict_track = {}

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

        pred_r, pred_t, pred_c, emb = self.estimator(
            img, points, choose, idx)

        return pred_r, pred_t, pred_c, emb

    def training_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0
        l = len(batch)
        for frame in batch:

            # unpack the batch and apply forward pass
            if frame[0].dtype == torch.bool:
                continue

            points, choose, img, target, model_points, idx = frame[0:6]
            ff_trans, ff_rot, depth_img, img_orig, cam = frame[6:11]
            gt_rot_wxyz, gt_trans, unique_desig = frame[11:14]

            pred_r, pred_t, pred_c, emb = self(img, points, choose, idx)

            loss, dis, new_points, new_target = self.criterion(
                pred_r, pred_t, pred_c, target, model_points, idx, points, self.w, self.refine)  # wxy
            total_loss += loss
            total_dis += dis
        # choose correct loss here
        total_loss = total_loss / l
        total_dis = total_dis.detach() / l
        tensorboard_logs = {'train_loss': total_loss, 'train_dis': total_dis}
        return {'loss': total_loss, 'dis': total_dis, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0

        for frame in batch:

            if frame[0].dtype == torch.bool:
                continue

            # unpack the batch and apply forward pass
            points, choose, img, target, model_points, idx = frame[0:6]
            ff_trans, ff_rot, depth_img, img_orig, cam = frame[6:11]
            gt_rot_wxyz, gt_trans, unique_desig = frame[11:14]

            pred_r, pred_t, pred_c, emb = self(img, points, choose, idx)

            loss, dis, new_points, new_target = self.criterion(
                pred_r, pred_t, pred_c, target, model_points, idx, points, self.w, self.refine)  # wxy
            if f'{int(unique_desig[1])}_loss' in self._dict_track.keys():
                self._dict_track[f'{int(unique_desig[1])}_loss'].append(
                    float(loss))
                self._dict_track[f'{int(unique_desig[1])}_dis'].append(
                    float(dis))
            else:
                self._dict_track[f'{int(unique_desig[1])}_loss'] = [
                    float(loss)]
                self._dict_track[f'{int(unique_desig[1])}_dis'] = [float(dis)]

            total_loss += loss
            total_dis += dis

        tensorboard_logs = {'val_loss': total_loss /
                            len(batch), 'val_dis': total_dis / len(batch)}
        return {'val_loss': total_loss / len(batch), 'val_dis': total_dis / len(batch), 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0
        for frame in batch:

            if frame[0].dtype == torch.bool:
                continue

            # unpack the batch and apply forward pass
            points, choose, img, target, model_points, idx = frame[0:6]
            ff_trans, ff_rot, depth_img, img_orig, cam = frame[6:11]
            gt_rot_wxyz, gt_trans, unique_desig = frame[11:14]

            pred_r, pred_t, pred_c, emb = self(img, points, choose, idx)

            loss, dis, new_points, new_target = self.criterion(
                pred_r, pred_t, pred_c, target, model_points, idx, points, self.w, self.refine)  # wxy

            self.Visu.plot_estimated_pose(tag='ground_truth_%d' % (batch_idx),
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

            total_loss += loss
            total_dis += dis

        tensorboard_logs = {'val_loss': total_loss /
                            len(batch), 'val_dis': total_dis / len(batch)}
        return {'val_loss': total_loss / len(batch), 'val_dis': total_dis / len(batch), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        self._df = pd.DataFrame.from_dict(self._dict_track)
        self._dict_track = {}

        val_dis_mean = 0
        for output in outputs:
            val_dis = output['val_dis']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_dis = torch.mean(val_dis)
            val_dis_mean += val_dis
        val_dis_mean /= len(outputs)

        if val_dis_mean < self.best_validation:

            self.best_validation = val_dis_mean
            self.best_validation_patience_run = 0
        else:
            self.best_validation_patience_run += 1
            if self.best_validation_patience_run > self.best_validation_patience:
                print("figure out how to set stop training flag")

        if val_dis_mean < self.early_stopping_value:
            print("figure out how to set stop training flag")

        tensorboard_logs = {'val_dis_epoch': float(val_dis_mean.detach())}
        tensorboard_logs.update(self._df.mean().to_dict())

        return {'val_dis_epoch': val_dis_mean.detach(), 'val_dis_epoch_float': float(val_dis_mean.detach()), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=0.0001)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, threshold=0.02),
            'monitor': 'val_dis_epoch',  # Default: val_loss
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset_train = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)
        dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                       batch_size=self.exp['loader']['batch_size'],
                                                       shuffle=True,
                                                       num_workers=self.exp['loader']['workers'],
                                                       pin_memory=True)
        return dataloader_train

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_val'],
            cfg_env=self.env)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=self.exp['loader']['batch_size'],
                                                      shuffle=False,
                                                      num_workers=self.exp['loader']['workers'],
                                                      pin_memory=True)
        return dataloader_test

    def val_dataloader(self):
        dataset_val = GenericDataset(
            cfg_d=self.exp['d_val'],
            cfg_env=self.env)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    args = parser.parse_args()
    exp_cfg_path = args.exp
    env_cfg_path = args.env

    exp = ConfigLoader().from_file(exp_cfg_path)
    env = ConfigLoader().from_file(env_cfg_path)
    # keep this in mind
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
    # TODO create one earlz stopping callback
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/63bd0582e35ad865c1f07f61975456f65de0f41f/pytorch_lightning/callbacks/base.py#L12
    early_stop_callback = EarlyStopping(
        monitor='val_dis_epoch_float',
        patience=4,
        strict=True,
        verbose=True,
        mode='min'
    )

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        verbose=True,
        monitor="val_dis_epoch",
        mode="min",
        prefix="",
    )

    from pytorch_lightning.logging import CometLogger

    # arguments made to CometLogger are passed on to the comet_ml.Experiment class

    trainer = Trainer(gpus=1,
                      num_nodes=1,
                      auto_lr_find=False,
                      accumulate_grad_batches=exp['accumulate_grad_batches'],
                      default_root_dir=model_path,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback,
                      fast_dev_run=True,
                      limit_train_batches=0.0005,
                      limit_test_batches=0.1,
                      limit_val_batches=0.01,
                      val_check_interval=100,
                      terminate_on_nan=True)

    # trainer.fit(model)
    trainer.test(model)
