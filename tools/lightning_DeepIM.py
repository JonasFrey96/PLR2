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

# network dense fusion
#from lib.loss import Loss
#from lib.loss_refiner import Loss_refine
#from lib.network import PoseNet, PoseRefineNet
#from lib.motion_network import MotionNetwork
#from lib.motion_loss import motion_loss
# dataset
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


def get_bb_from_depth(depth):
    bb_lsd = []
    for d in depth:
        masked_idx = (d != 0).nonzero()
        min1 = torch.min(masked_idx[:, 0]).type(torch.float32)
        max1 = torch.max(masked_idx[:, 0]).type(torch.float32)
        min2 = torch.min(masked_idx[:, 1]).type(torch.float32)
        max2 = torch.max(masked_idx[:, 1]).type(torch.float32)
        bb_lsd.append(BoundingBox(p1=torch.stack(
            [min1, min2]), p2=torch.stack([max1, max2])))
    return bb_lsd


def backproject_point(p, fx, fy, cx, cy):
    u = int(((p[0] / p[2]) * fx) + cx)
    v = int(((p[1] / p[2]) * fy) + cy)
    return u, v


def backproject_points(p, fx, fy, cx, cy):
    """
    p.shape = (nr_points,xyz)
    """
    # true_divide
    u = torch.round((torch.div(p[:, 0], p[:, 2]) * fx) + cx)
    v = torch.round((torch.div(p[:, 1], p[:, 2]) * fy) + cy)

    if torch.isnan(u).any() or torch.isnan(v).any():
        u = torch.tensor(cx).unsqueeze(0)
        v = torch.tensor(cy).unsqueeze(0)
        print('Predicted z=0 for translation. u=cx, v=cy')
        # raise Exception

    return torch.stack([v, u]).T


def backproject_points_batch(p, fx, fy, cx, cy):
    """
    p.shape = (nr_points,xyz)
    """
    bs, dim, _ = p.shape
    p = p.view(-1, 3)

    u = torch.round(torch.true_divide(p[:, 0], p[:, 2]).view(
        bs, -1) * fx.view(bs, -1).repeat(1, dim) + cx.view(bs, -1).repeat(1, dim))
    v = torch.round(torch.true_divide(p[:, 0], p[:, 1]).view(
        bs, -1) * fy.view(bs, -1).repeat(1, dim) + cy.view(bs, -1).repeat(1, dim))

    return torch.stack([u, v], dim=2)


def plt_img(img, name='plt_img.png', folder='/home/jonfrey/Debug'):

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(folder + '/' + name)


def plt_torch(data, name='torch.png', folder='/home/jonfrey/Debug'):
    img = render = np.transpose(
        data[:, :, :].cpu().numpy().astype(np.uint8), (2, 1, 0))
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(folder + '/' + name)


def visu_network_input(data, folder='/home/jonfrey/Debug', max_images=10):
    num = min(max_images, data.shape[0])
    fig = plt.figure(figsize=(7, num * 3.5))

    for i in range(num):

        n_render = f'batch{i}_render.png'
        n_real = f'batch{i}_real.png'
        real = np.transpose(
            data[i, :3, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))
        render = np.transpose(
            data[i, 3:, :, :].cpu().numpy().astype(np.uint8), (1, 2, 0))

        # plt_img(real, name=n_real, folder=folder)
        # plt_img(render, name=n_render, folder=folder)

        fig.add_subplot(num, 2, i * 2 + 1)
        plt.imshow(real)
        plt.tight_layout()
        fig.add_subplot(num, 2, i * 2 + 2)
        plt.imshow(render)
        plt.tight_layout()
    plt.savefig(folder + '/complete_batch.png', dpi=300)


def visu_projection(target, images, cam, folder='/home/jonfrey/Debug', max_images=10):
    num = min(max_images, target.shape[0])
    fig = plt.figure(figsize=(7, num * 3.5))
    for i in range(num):
        masked_idx = backproject_points(
            target[i], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])

        for j in range(masked_idx.shape[0]):
            try:
                images[i, int(masked_idx[j, 0]), int(masked_idx[j, 1]), 0] = 0
                images[i, int(masked_idx[j, 0]), int(
                    masked_idx[j, 1]), 1] = 255
                images[i, int(masked_idx[j, 0]), int(masked_idx[j, 1]), 2] = 0
            except:
                pass

        min1 = torch.min(masked_idx[:, 0])
        max1 = torch.max(masked_idx[:, 0])
        max2 = torch.max(masked_idx[:, 1])
        min2 = torch.min(masked_idx[:, 1])

        bb = BoundingBox(p1=torch.stack(
            [min1, min2]), p2=torch.stack([max1, max2]))

        print(f'Bounding box is valid {bb.valid()}, {str(bb)}')

        bb_img = bb.plot(images[i, :, :, :3].cpu().numpy().astype(np.uint8))
        fig.add_subplot(num, 2, i * 2 + 1)
        plt.imshow(bb_img)

        fig.add_subplot(num, 2, i * 2 + 2)
        real = images[i, :, :, :3].cpu().numpy().astype(np.uint8)
        plt.imshow(real)

    plt.savefig(folder + '/project_batch.png', dpi=300)


def get_bb_real_target(target, cam, gt_trans):
    bb_ls = []
    # ret = backproject_points_batch(
    #     target, fx=cam[:, 2], fy=cam[:, 3], cx=cam[:, 0], cy=cam[:, 1])
    # min_val, min_ind = torch.min(ret, dim=1, keepdim=True)
    # max_val, max_ind = torch.max(ret, dim=1, keepdim=True)

    for i in range(target.shape[0]):
        # could not find a smart alternative to avoide looping
        masked_idx = backproject_points(
            target[i], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])
        min1 = torch.min(masked_idx[:, 0])
        max1 = torch.max(masked_idx[:, 0])
        max2 = torch.max(masked_idx[:, 1])
        min2 = torch.min(masked_idx[:, 1])

        bb = BoundingBox(p1=torch.stack(
            [min1, min2]), p2=torch.stack([max1, max2]))

        # val_image = img_orig[i, :, :, :].cpu().numpy()
        # bb.plot(val_image)

        # center_real = backproject_point(
        # gt_trans[i, :], fx=cam[i, 2], fy=cam[i, 3], cx=cam[i, 0], cy=cam[i, 1])
        # bb.move(-center_real[1], -center_real[0])
        bb_ls.append(bb)

    return bb_ls


class TrackNet6D(LightningModule):
    def __init__(self, exp, env):
        super().__init__()
        self.vm = None
        self.nr_of_images_per_object = exp.get(
            'vm_nr_of_images_per_object', 10)
        # logging h-params
        exp_config_flatten = flatten_dict(copy.deepcopy(exp))
        for k in exp_config_flatten.keys():
            if exp_config_flatten[k] is None:
                exp_config_flatten[k] = 'is None'

        self.hparams = exp_config_flatten
        self.hparams['lr'] = exp['lr']

        self.test_size = 0.9
        self.env, self.exp = env, exp

        restore_deepim_refiner = '/media/scratch1/jonfrey/models/pretrained_flownet/FlowNetModels/pytorch/flownets_from_caffe.pth.tar'

        self.refiner = DeepIM.from_weights(
            21, restore_deepim_refiner)

        num_poi = exp['d_train']['num_pt_mesh_small']
        #self.criterion = Loss(num_poi, exp['d_train']['obj_list_sym'])
        num_poi = exp['d_train']['num_pt_mesh_large']
        # self.criterion_refine = Loss_refine(
        #    num_poi, exp['d_train']['obj_list_sym'])

        self.criterion_adds = LossAddS(sym_list=exp['d_train']['obj_list_sym'])

        self.refine = False
        self.w = exp['w_normal']
        self.w_decayed = False

        self.best_validation = 999
        self.best_validation_patience = 5
        self.early_stopping_value = 0.1

        self.Visu = None
        self._dict_track = {}

        self.number_images_log_val = 5
        self.number_images_log_test = 10
        self.counter_images_logged = 0

        self.init_train_vali_split = False

    def forward(self, batch):

        # Hyper parameters that should  be moved to config
        refine_iterations = self.exp.get(
            'training', {}).get('refine_iterations', 1)
        translation_noise = 0.03
        # unpack batch
        points, choose, img, target, model_points, idx = batch[0:6]
        depth_img, label_img, img_orig, cam = batch[6:10]
        gt_rot_wxyz, gt_trans, unique_desig = batch[10:13]

        # start with an inital guess (here we take the noisy version later from the dataloader or replay buffer implementation)
        # current estimate of the rotation and translations

        pred_rot_wxyz, pred_trans = gt_rot_wxyz, torch.normal(
            mean=gt_trans, std=self.exp.get('training', {}).get('translation_noise_inital', 0.03))
        # pred_rot_mat = quat_to_rot(
        # pred_rot_wxyz, conv='wxyz', device=self.device)

        base_inital = quat_to_rot(
            pred_rot_wxyz, 'wxyz', device=self.device).unsqueeze(1)
        base_inital = base_inital.view(-1, 3, 3).permute(0,
                                                         2, 1)  # transposed of R

        pred_points = torch.add(
            torch.bmm(model_points, base_inital), pred_trans.unsqueeze(1))

        # current estimate of the object points
        # rot_mat = pred_rot_mat.unsqueeze(1).repeat(
        # (1, model_points.shape[1], 1, 1)).view(-1, 3, 3)

        w = 640
        h = 480
        bs = img.shape[0]

        for i in range(0, refine_iterations):

            render_img = torch.empty((bs, 3, h, w), device=self.device)
            # preper render data
            st = time.time()
            img_ren, depth, h_render = self.vm.get_closest_image_batch(
                i=idx, rot=pred_rot_wxyz, conv='wxyz')

            # print(f'Total Time VM blocking {time.time()-st}')
            bb_lsd = get_bb_from_depth(depth)
            for j, b in enumerate(bb_lsd):
                center_ren = backproject_points(
                    h_render[j, :3, 3].view(1, 3), fx=cam[j, 2], fy=cam[j, 3], cx=cam[j, 0], cy=cam[j, 1])
                center_ren = center_ren.squeeze()
                b.move(-center_ren[1], -center_ren[0])
                b.expand(1.1)
                b.expand_to_correct_ratio(w, h)
                b.move(center_ren[1], center_ren[0])
                crop_ren = b.crop(img_ren[j]).unsqueeze(0)
                up = torch.nn.UpsamplingBilinear2d(size=(h, w))
                crop_ren = torch.transpose(crop_ren, 1, 3)
                crop_ren = torch.transpose(crop_ren, 2, 3)
                render_img[j] = up(crop_ren)

            # prepare real data
            real_img = torch.empty((bs, 3, h, w), device=self.device)
            # update the target to get new bb
            bb_ls = get_bb_real_target(pred_points, cam, pred_trans)
            for j, b in enumerate(bb_ls):

                center_real = backproject_points(
                    pred_trans[j].view(1, 3), fx=cam[j, 2], fy=cam[j, 3], cx=cam[j, 0], cy=cam[j, 1])
                center_real = center_real.squeeze()
                b.move(-center_real[0], -center_real[1])
                b.expand(1.1)
                b.expand_to_correct_ratio(w, h)
                b.move(center_real[0], center_real[1])
                crop_real = b.crop(img_orig[j]).unsqueeze(0)
                up = torch.nn.UpsamplingBilinear2d(size=(h, w))
                crop_real = torch.transpose(crop_real, 1, 3)
                crop_real = torch.transpose(crop_real, 2, 3)
                real_img[j] = up(crop_real)

            # stack the two images, might add additional mask as layer or depth info
            data = torch.cat([real_img, render_img], dim=1)

            # visu_network_input(data)
            # visu_projection(pred_points, img_orig, cam,
            # folder='/home/jonfrey/Debug', max_images=5)
            f2, f3, f4, f5, f6, delta_v, delta_rot_wxyz = self.refiner(
                data, idx)

            # Update translation prediction
            pred_trans_new = get_delta_t_in_euclidean(
                delta_v.clone(), t_src=pred_trans.clone(),
                fx=cam[:, 2].unsqueeze(1), fy=cam[:, 3].unsqueeze(1), device=self.device)

            if torch.isnan(pred_trans_new).any():
                print(pred_trans_new, pred_trans, delta_v)
                assert Exception

            delta_t = pred_trans_new - pred_trans
            # delta_t can be used for bookkeeping to keep track of rotation
            pred_trans = pred_trans_new

            # Update rotation prediction
            pred_rot_wxyz = compose_quat(pred_rot_wxyz, delta_rot_wxyz)

            # Update pred_points
            base_new = quat_to_rot(
                pred_rot_wxyz, 'wxyz', device=self.device).unsqueeze(1)
            base_new = base_new.view(-1, 3, 3).permute(0,
                                                       2, 1)  # transposed of R
            pred_points = torch.add(
                torch.bmm(model_points, base_new), pred_trans.unsqueeze(1))

        # Compute ADD / ADD-S loss
        dis = self.criterion_adds(pred_r=pred_rot_wxyz, pred_t=pred_trans,
                                  target=target, model_points=model_points, idx=idx)

        return dis, pred_rot_wxyz, pred_trans

    def training_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0

        st = time.time()
        dis, _, _ = self(batch[0])
        loss = torch.sum(dis)
        print(time.time() - st)
        # for epoch average logging
        try:
            self._dict_track['train_loss'].append(float(loss))
        except:
            self._dict_track['train_loss'] = [float(loss)]

        # tensorboard logging
        tensorboard_logs = {'train_loss': float(loss)}
        #                     'train_loss_without_motion': float(0)}
        # 'dis': total_dis, 'log': tensorboard_logs, 'progress_bar': {'train_dis': total_dis, 'train_loss': total_loss}}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        total_loss = 0
        total_dis = 0

        st = time.time()
        dis, pred_r, pred_t = self(batch[0])
        loss = torch.sum(dis)

        if self.counter_images_logged < self.number_images_log_val:
            self.counter_images_logged += 1
            points, choose, img, target, model_points, idx = batch[0][0:6]
            depth_img, label_img, img_orig, cam = batch[0][6:10]
            gt_rot_wxyz, gt_trans, unique_desig = batch[0][10:13]
            self.visu_pose(batch_idx, pred_r[0], pred_t[0],
                           target[0], model_points[0], cam[0], img_orig[0], unique_desig, idx[0])
            # for epoch average logging
        try:
            self._dict_track['val_loss'].append(float(loss))
        except:
            self._dict_track['val_loss'] = [float(loss)]

        tensorboard_logs = {'val_loss': float(loss)}
        val_loss = loss
        val_dis = loss
        return {'val_loss': val_loss, 'val_dis': val_dis}

    def validation_epoch_end(self, outputs):
        avg_dict = {}
        # only keys that are logged in tensorboard are removed from log_dict !
        for old_key in list(self._dict_track.keys()):
            if old_key.find('val') == -1:
                continue
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
            self._dict_track.pop(old_key, None)

        self.counter_images_logged = 0  # reset image log counter

        avg_val_dis_float = 1
        return {'avg_val_dis_float': avg_val_dis_float, 'log': avg_dict}

    def train_epoch_end(self, outputs):
        avg_dict = {}
        for old_key in list(self._dict_track.keys()):
            if old_key.find('train') == -1:
                continue
            avg_dict['avg_' +
                     old_key] = float(np.mean(np.array(self._dict_track[old_key])))
            self._dict_track.pop(old_key, None)

        return {
            **avg_dict, 'log': avg_dict}

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

    def visu_pose(self, batch_idx, pred_r, pred_t, target, model_points, cam, img_orig, unique_desig, idx):
        if self.Visu is None:
            self.Visu = Visualizer(exp['model_path'] +
                                   '/visu/', self.logger.experiment)

        self.Visu.plot_estimated_pose(tag='ground_truth_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                      epoch=self.current_epoch,
                                      img=img_orig.detach().cpu().numpy(),
                                      points=copy.deepcopy(
            target.detach().cpu().numpy()),
            trans=np.array([[0, 0, 0]]),
            rot_mat=np.diag((1, 1, 1)),
            cam_cx=float(cam[0]),
            cam_cy=float(cam[1]),
            cam_fx=float(cam[2]),
            cam_fy=float(cam[3]),
            store=False)
        # extract highest confident vote

        t = pred_t.detach().cpu().numpy()
        r = pred_r.detach().cpu().numpy()

        rot = R.from_quat(re_quat(r, 'wxyz'))

        self.Visu.plot_estimated_pose(tag='final_prediction_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                      epoch=self.current_epoch,
                                      img=img_orig[:, :,
                                                   :].detach().cpu().numpy(),
                                      points=copy.deepcopy(
                                          model_points[:, :].detach(
                                          ).cpu().numpy()),
                                      trans=t.reshape((1, 3)),
                                      rot_mat=rot.as_matrix(),
                                      cam_cx=float(cam[0]),
                                      cam_cy=float(cam[1]),
                                      cam_fx=float(cam[2]),
                                      cam_fy=float(cam[3]),
                                      store=False)
        img_ren, depth, h_render = self.vm.get_closest_image_batch(
            i=idx.unsqueeze(0), rot=pred_r.unsqueeze(0), conv='wxyz')
        test = 999

        self.Visu.plot_estimated_pose(tag='render_with_gt_%s_obj%d' % (str(unique_desig[0][0]).replace('/', "_"), int(unique_desig[1][0])),
                                      epoch=self.current_epoch,
                                      img=img_ren[0, :, :,
                                                  :].detach().cpu().numpy(),
                                      points=copy.deepcopy(
            target.detach().cpu().numpy()),
            trans=np.array([[0, 0, 0]]),
            rot_mat=np.diag((1, 1, 1)),
            cam_cx=float(cam[0]),
            cam_cy=float(cam[1]),
            cam_fx=float(cam[2]),
            cam_fy=float(cam[3]),
            store=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.refiner.parameters()}], lr=self.hparams['lr'])
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
                                                       batch_size=self.exp[
                                                           'loader']['batch_size'],
                                                       shuffle=True,
                                                       num_workers=self.exp[
                                                           'loader']['workers'],
                                                       pin_memory=True)

        store = self.env['p_ycb'] + '/viewpoints_renderings'
        if self.vm is None:
            self.vm = ViewpointManager(
                store, dataset_train._backend._name_to_idx, nr_of_images_per_object=self.nr_of_images_per_object, device=self.device, load_images=False)

        return dataloader_train

    # def test_dataloader(self):
    #     dataset_test = GenericDataset(
    #         cfg_d=self.exp['d_test'],
    #         cfg_env=self.env)

    #     dataloader_test = torch.utils.data.DataLoader(dataset_test,
    #                                                   batch_size=self.exp['loader']['batch_size'],
    #                                                   shuffle=False,
    #                                                   num_workers=self.exp['loader']['workers'],
    #                                                   pin_memory=True)
    #     return dataloader_test

    def val_dataloader(self):
        dataset_val = GenericDataset(
            cfg_d=self.exp['d_train'],
            cfg_env=self.env)

        store = self.env['p_ycb'] + '/viewpoints_renderings'
        if self.vm is None:
            self.vm = ViewpointManager(
                store,
                dataset_val._backend._name_to_idx,
                nr_of_images_per_object=self.nr_of_images_per_object,
                device=self.device,
                load_images=False)

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


if __name__ == "__main__":
    # for reproducability
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=file_path, default='yaml/exp/exp_ws_deepim.yml',  # required=True,
                        help='The main experiment yaml file.')
    parser.add_argument('--env', type=file_path, default='yaml/env/env_natrix_jonas.yml',
                        help='The environment yaml file.')
    args = parser.parse_args()
    exp_cfg_path = args.exp
    env_cfg_path = args.env

    exp = ConfigLoader().from_file(exp_cfg_path).get_FullLoader()
    env = ConfigLoader().from_file(env_cfg_path).get_FullLoader()

    """
    Trainer args(gpus, num_nodes, etc…) & & Program arguments(data_path, cluster_email, etc…)
    Model specific arguments(layer_dim, num_layers, learning_rate, etc…)
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
        patience=exp.get('early_stopping_cfg', {}).get('patience', 100),
        strict=True,
        verbose=True,
        mode='min'
    )

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=exp['model_path'] + '/{epoch}-{avg_val_dis:.4f}',
        verbose=True,
        monitor="avg_val_dis",
        mode="min",
        prefix="",
        save_last=True,
        save_top_k=10,
    )
    if exp.get('checkpoint_restore', False):
        checkpoint = torch.load(
            exp['checkpoint_path'], map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    with torch.autograd.set_detect_anomaly(True):
        trainer = Trainer(gpus=1,
                          num_nodes=1,
                          auto_lr_find=False,
                          accumulate_grad_batches=exp['training']['accumulate_grad_batches'],
                          default_root_dir=model_path,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop_callback,
                          fast_dev_run=False,
                          limit_train_batches=200,
                          limit_val_batches=1000,
                          limit_test_batches=1.0,
                          val_check_interval=1.0,
                          progress_bar_refresh_rate=exp['training']['accumulate_grad_batches'],
                          max_epochs=exp['training']['max_epochs'],
                          terminate_on_nan=False)

    if exp.get('model_mode', 'fit') == 'fit':
        # lr_finder = trainer.lr_find(
        #     model, min_lr=0.0000001, max_lr=0.001, num_training=500, early_stop_threshold=100)
        # Results can be found in
        # lr_finder.results
        # lr_finder.suggestion()
        trainer.fit(model)
    elif exp.get('model_mode', 'fit') == 'test':
        trainer.test(model)
        if exp.get('conv_test2df', False):
            command = 'cd scripts & python experiment2df.py %s --write-csv --write-pkl' % (
                model_path + '/lightning_logs/version_0')
            os.system(command)
    else:
        print("Wrong model_mode defined in exp config")
        raise Exception
