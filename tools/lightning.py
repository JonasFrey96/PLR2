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
import random
import sklearn
from scipy.spatial.transform import Rotation as R
import datetime

sys.path.insert(0, os.getcwd())
print(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))
# src modules
from helper import pad
from loaders_v2 import ConfigLoader

import torch
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.loss import KeypointLoss, MultiObjectADDLoss
from lib import keypoint_helper as kp_helper
from lib.network import KeypointNet

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

        self.validation_size = 0.05
        self.env, self.exp = env, exp

        self.estimator = KeypointNet(**exp['net'])

        if exp['estimator_restore']:
            state_dict = torch.load(exp['estimator_load'])
            self.load_my_state_dict(state_dict)

        self.criterion = KeypointLoss(**exp['loss'])
        self.add_loss = MultiObjectADDLoss()

        self.visualizer = None
        self._dict_track = {}

        self.number_images_log_val = 5
        self.number_images_log_test = 10
        self.counter_images_logged = 0

        self._init_datasets()

    def _init_datasets(self):
        dataset_train = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        self.indices_train, self.indices_valid = sklearn.model_selection.train_test_split(
            range(0, len(dataset_train)), test_size=self.validation_size)
        self.dataset_train = torch.utils.data.Subset(
            dataset_train, self.indices_train)

        self.dataset_val = torch.utils.data.Subset(dataset_train, self.indices_valid)
        self.object_models = dataset_train.object_models()
        self.keypoints = dataset_train.keypoints()
        self.K = self.keypoints.shape[1]

    def load_my_state_dict(self, checkpoint_state):
        state_dict = checkpoint_state['state_dict']
        state = {}
        for key, value in state_dict.items():
            if key.index('estimator.') == 0:
                key = key.replace('estimator.', '')
                state[key] = value
        self.estimator.load_state_dict(state)

    def forward(self, img, points):
        return self.estimator(img, points)

    def training_step(self, batch, batch_idx):
        total_loss = 0
        l = len(batch)
        keypoint_loss = 0
        center_loss = 0
        semantic_loss = 0
        for frame in batch:
            # unpack the batch and apply forward pass
            if frame[0].dtype == torch.bool:
                continue

            (points, img, label, gt_keypoints, gt_centers, cam,
                    objects_in_scene, unique_desig) = frame

            predicted_keypoints, object_centers, segmentation = self(img, points)
            loss, losses = self.criterion(predicted_keypoints, object_centers, segmentation,
                    gt_keypoints, gt_centers, label)
            total_loss += loss

            keypoint_loss += losses[0]
            center_loss += losses[1]
            semantic_loss += losses[2]

        total_loss = total_loss / l
        tensorboard_logs = {'train_loss': total_loss,
                'keypoint_loss': keypoint_loss.item(),
                'center_loss': center_loss.item(),
                'semantic_loss': semantic_loss.item()}
        return {'loss': total_loss, 'keypoint_loss': keypoint_loss, 'center_loss': center_loss,
                'semantic_loss': semantic_loss, 'log': tensorboard_logs,
                'progress_bar': {'train_loss': total_loss, 'kp_loss': keypoint_loss, 's_loss': semantic_loss}}

    def validation_step(self, batch, batch_idx):
        total_loss = 0
        keypoint_loss = 0
        center_loss = 0
        semantic_loss = 0

        model_keypoints = self.keypoints.to(self.device)
        object_models = self.object_models.to(self.device)

        for frame in batch:

            if frame[0].dtype == torch.bool:
                continue

            (points, img, label, gt_keypoints, gt_centers, cam,
                    objects_in_scene, unique_desig) = frame

            predicted_keypoints, object_centers, segmentation = self(img, points)
            loss, losses = self.criterion(predicted_keypoints, object_centers, segmentation,
                    gt_keypoints, gt_centers, label)

            N, _, H, W = gt_keypoints.shape
            predicted_keypoints = predicted_keypoints.reshape(N, self.K, 3, H, W)
            gt_keypoints = gt_keypoints.reshape(N, self.K, 3, H, W)

            if 'val_loss' not in self._dict_track:
                self._dict_track['val_loss'] = []
            self._dict_track['val_loss'].append(loss)

            if self.number_images_log_val > self.counter_images_logged:
                self.visualize(batch_idx, predicted_keypoints, object_centers, segmentation, points, label, gt_keypoints,
                        gt_centers, cam, img, unique_desig)
                self.counter_images_logged += 1

            keypoint_loss += losses[0]
            center_loss += losses[1]
            semantic_loss += losses[2]
            total_loss += loss

        tensorboard_logs = {'val_loss': total_loss / len(batch),
                'keypoint_loss': keypoint_loss.item(),
                'center_loss': center_loss.item(),
                'semantic_loss': semantic_loss.item()}
        return {'val_loss': total_loss / len(batch),
                'keypoint_loss': keypoint_loss,
                'center_loss': center_loss,
                'semantic_loss': semantic_loss,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # used for tensorboard logging
        total_loss = 0
        keypoint_loss = 0
        center_loss = 0
        semantic_loss = 0

        for frame in batch:

            if frame[0].dtype == torch.bool:
                continue

            (points, img, label, gt_keypoints, gt_centers, cam, objects_in_scene, unique_desig) = frame
            predicted_keypoints, object_centers, segmentation = self(img, points)
            loss, losses = self.criterion(predicted_keypoints, object_centers, segmentation,
                    gt_keypoints, gt_centers, label)

            # self._dict_track is used to log whole epoch
            if 'test_loss' not in self._dict_track:
                self._dict_track['test_loss'] = []
            self._dict_track['test_loss'].append(loss)

            if self.number_images_log_test > self.counter_images_logged:
                N, _, H, W = gt_keypoints.shape
                predicted_keypoints = predicted_keypoints.reshape(N, self.K, 3, H, W)
                gt_keypoints = gt_keypoints.reshape(N, self.K, 3, H, W)
                self.visualize(batch_idx, predicted_keypoints, object_centers, segmentation, points, label, gt_keypoints,
                        gt_centers, cam, img, unique_desig)
                self.counter_images_logged += 1

            keypoint_loss += losses[0]
            center_loss += losses[1]
            semantic_loss += losses[2]
            total_loss += loss

        tensorboard_logs = {'test_loss': total_loss / len(batch),
                'keypoint_loss': keypoint_loss,
                'center_loss': center_loss,
                'semantic_loss': semantic_loss}

        return {**tensorboard_logs,
                'keypoint_loss': keypoint_loss,
                'center_loss': center_loss,
                'semantic_loss': semantic_loss,
                'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_dict = {}
        for key, values in self._dict_track.items():
            avg_dict['avg_' + key] = torch.stack(values).mean()
        self._dict_track = {}

        return {
            **avg_dict, 'log': avg_dict}

    def validation_epoch_end(self, outputs):
        avg_dict = {}
        for key, values in self._dict_track.items():
            avg_dict['avg_' + key] = torch.stack(values, dim=0).mean().detach().cpu()
        self._dict_track = {}

        self.counter_images_logged = 0  # reset image log counter

        tensorboard_log = copy.deepcopy(avg_dict)
        for key, value in tensorboard_log.items():
            tensorboard_log[key] = value.item()

        return {**avg_dict, 'log': tensorboard_log}

    def visualize(self, batch_idx, predicted_keypoints, predicted_centers, predicted_label,
            points, label, gt_keypoints, gt_centers, cam, img_orig, unique_desig):
        img_orig = img_orig.cpu().numpy()
        img_orig = ((img_orig + 1.0) * 127.5).astype(np.uint8)
        img_orig = img_orig.transpose([0, 2, 3, 1])
        predicted_keypoints = predicted_keypoints.transpose(1, 3).transpose(2, 4)
        gt_keypoints = gt_keypoints.transpose(1, 3).transpose(2, 4)
        gt_centers = gt_centers.transpose(1, 3).transpose(1, 2)
        points = points.transpose(1, 3).transpose(1, 2)
        if self.visualizer is None:
            self.visualizer = Visualizer(os.path.join(exp['model_path'], 'visu'), self.logger.experiment)

        random_index = np.random.randint(0, points.shape[0])
        self.visualizer.plot_keypoints(tag='gt_keypoints_{}'.format(str(unique_desig[random_index]).replace('/', "_")),
            epoch=self.current_epoch,
            img=img_orig[random_index, :, :, :],
            points=points[random_index].cpu().numpy(),
            keypoints=gt_keypoints[random_index].cpu().numpy(),
            label=label[random_index].cpu().numpy(),
            cam_cx=float(cam[random_index, 0]),
            cam_cy=float(cam[random_index, 1]),
            cam_fx=float(cam[random_index, 2]),
            cam_fy=float(cam[random_index, 3]),
            store=True)

        self.visualizer.plot_keypoints(tag='predicted_{}'.format(unique_desig[random_index].replace('/', '_')),
                epoch=self.current_epoch,
                img=img_orig[random_index, :, :, :],
                points=points[random_index].cpu().numpy(),
                keypoints=predicted_keypoints[random_index].detach().cpu().numpy(),
                label=label[random_index].cpu().numpy(),
                cam_cx=cam[random_index, 0].item(),
                cam_cy=cam[random_index, 1].item(),
                cam_fx=cam[random_index, 2].item(),
                cam_fy=cam[random_index, 3].item(),
                store=True)

        self.visualizer.plot_centers(tag='gt_centers_{}'.format(unique_desig[random_index]).replace('/', '_'),
                epoch=self.current_epoch,
                img=img_orig[random_index, :, :, :],
                points=points[random_index].cpu().numpy(),
                centers=gt_centers[random_index].cpu().numpy(),
                label=label[random_index].cpu().numpy(),
                cam_cx=cam[random_index, 0].item(),
                cam_cy=cam[random_index, 1].item(),
                cam_fx=cam[random_index, 2].item(),
                cam_fy=cam[random_index, 3].item(),
                store=True)

        predicted_centers = predicted_centers[random_index].cpu().transpose(0, 1).transpose(1, 2).numpy()
        self.visualizer.plot_centers(tag='predicted_centers_{}'.format(unique_desig[random_index]).replace('/', '_'),
                epoch=self.current_epoch,
                img=img_orig[random_index, :, :, :],
                points=points[random_index].cpu().numpy(),
                centers=predicted_centers,
                label=label[random_index].cpu().numpy(),
                cam_cx=cam[random_index, 0].item(),
                cam_cy=cam[random_index, 1].item(),
                cam_fx=cam[random_index, 2].item(),
                cam_fy=cam[random_index, 3].item(),
                store=True)

        label = label[random_index]
        self.visualizer.plot_segmentation(tag='gt_segmentation_{}'.format(unique_desig[random_index].replace('/', '_')),
                epoch=self.current_epoch,
                label=label.cpu().numpy(),
                store=True)

        predicted_label = predicted_label[random_index].argmax(dim=0)
        self.visualizer.plot_segmentation(tag='predicted_segmentation_{}'.format(unique_desig[random_index].replace('/', '_')),
                epoch=self.current_epoch,
                label=predicted_label.cpu().numpy(),
                store=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.estimator.parameters(), lr=self.exp['lr'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                verbose=True,
                **self.exp['lr_cfg']['on_plateau_cfg']),
            'monitor': 'avg_val_loss',  # Default: val_loss
            'interval': self.exp['lr_cfg']['interval'],
            'frequency': self.exp['lr_cfg']['frequency']
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_train,
                                                       batch_size=self.exp['loader']['batch_size'],
                                                       shuffle=True,
                                                       num_workers=self.exp['loader']['workers'],
                                                       pin_memory=True)

    def test_dataloader(self):
        dataset_test = GenericDataset(
            cfg_d=self.exp['d_test'],
            cfg_env=self.env)

        return torch.utils.data.DataLoader(dataset_test,
                                                      batch_size=self.exp['loader']['batch_size'],
                                                      shuffle=False,
                                                      num_workers=self.exp['loader']['workers'],
                                                      pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_val,
                                                     batch_size=self.exp['loader']['batch_size'],
                                                     shuffle=False,
                                                     num_workers=self.exp['loader']['workers'],
                                                     pin_memory=True)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help="Run all steps quickly. Used for development runs.")
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    parser.add_argument('--gpus', type=int, default=1, help="How many gpus to use in training.")
    parser.add_argument('-w', '--workers', default=None)
    parser.add_argument('--ycb', default=None, help="Override the ycb video dataset path (e.g. if setting it from a variable).")
    parser.add_argument('--fp16', action='store_true', help='16-bit precision training')
    return parser.parse_args()

if __name__ == "__main__":
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

    if args.ycb is not None:
        env['p_ycb'] = args.ycb

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
        monitor='avg_val_loss',
        patience=exp.get('early_stopping_cfg', {}).get('patience', 10),
        strict=True,
        verbose=True,
        mode='min'
    )

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        verbose=True,
        monitor="avg_val_loss",
        mode="min",
        prefix="",
    )

    trainer = Trainer(gpus=args.gpus,
            precision=16 if args.fp16 else 32,
            default_root_dir=model_path,
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback,
            distributed_backend='ddp' if args.gpus > 1 else None,
            accumulate_grad_batches=exp.get('accumulate_grad', 1),
            fast_dev_run=args.dev)

    trainer.fit(model)
    trainer.test(model)
