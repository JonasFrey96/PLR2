from helper import compose_quat, rotation_angle, re_quat
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch
import time
import random
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from os import path
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import torch.utils.data as data
from PIL import Image
import string
import math
import coloredlogs
import logging
import os
import sys
import pickle
from estimation.state import State_R3xQuat, State_SE3, points
from helper import compose_quat, rotation_angle, re_quat
from visu import plot_pcd, plot_two_pcd
from helper import generate_unique_idx
from loaders_v2 import Backend, ConfigLoader
from helper import flatten_dict, get_bbox_480_640


class YCB(Backend):
    def __init__(self, cfg_d, cfg_env):
        super(YCB, self).__init__(cfg_d, cfg_env)
        self._cfg_d = cfg_d
        self._cfg_env = cfg_env
        self._p_ycb = cfg_env['p_ycb']
        self._pcd_cad_dict, self._name_to_idx = self.get_pcd_cad_models()
        self._batch_list = self.get_batch_list()

        self._length = len(self._batch_list)
        self._norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self._front_num = 2
        self._minimum_num_pt = 50
        self._xmap = np.array([[j for i in range(640)] for j in range(480)])
        self._ymap = np.array([[i for i in range(640)] for j in range(480)])

    def getElement(self, desig, obj_idx):
        """
        desig : sequence/idx
        two problems we face. What is if an object is not visible at all -> meta['obj'] = None
        """

        try:
            img = Image.open(
                '{0}/{1}-color.png'.format(self._p_ycb, desig))
            depth = np.array(Image.open(
                '{0}/{1}-depth.png'.format(self._p_ycb, desig)))
            label = np.array(Image.open(
                '{0}/{1}-label.png'.format(self._p_ycb, desig)))
            meta = scio.loadmat(
                '{0}/{1}-meta.mat'.format(self._p_ycb, desig))

        except:
            logging.error(
                'cant find files for {0}/{1}'.format(self._p_ycb, desig))
            return False

        cam = self.get_camera(desig)

        if self._cfg_d['output_cfg']['visu']['return_img']:
            img_copy = np.array(img.convert("RGB"))

        # what is this doing again
        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
        add_front = False

        # TODO add here correct way to load noise
        if self._cfg_d['noise_cfg']['status']:
            for k in range(5):

                seed = random.choice(self._syn)

                front = np.array(self._trancolor(Image.open(
                    '{0}/{1}-color.png'.format(self._p_ycb, desig)).convert("RGB")))

                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open(
                    '{0}/{1}-label.png'.format(self._p_ycb, seed)))

                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self._front_num:
                    continue
                front_label = random.sample(front_label, self._front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk

                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, obj_idx))
        mask = mask_label * mask_depth

        obj_idx_in_list = int(np.argwhere(obj == obj_idx))
        target_r = meta['poses'][:, :, obj_idx_in_list][:, 0:3]
        target_t = np.array(
            [meta['poses'][:, :, obj_idx_in_list][:, 3:4].flatten()])

        #gt_trans = copy.deepcopy(target_t[0, :])
        #gt_rot = re_quat(R.from_matrix(target_r).as_quat(), 'xyzw')
        if self._cfg_d['noise_cfg']['status']:
            add_t = np.array(
                [random.uniform(-self._cfg_d['noise_cfg']['noise_trans'], self._cfg_d['noise_cfg']['noise_trans']) for i in range(3)])
        else:
            add_t = np.zeros(3)

        gt_rot_wxyz = re_quat(
            R.from_matrix(target_r).as_quat(), 'xyzw')
        gt_trans = np.squeeze(target_t + add_t, 0)
        unique_desig = (desig, obj_idx)

        if len(mask.nonzero()[0]) <= self._minimum_num_pt:
          return (False, gt_rot_wxyz, gt_trans, unique_desig)

        # take the noise color image
        if self._cfg_d['noise_cfg']['status']:
            img = self._trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox_480_640(mask_label)
        # return the pixel coordinate for the bottom left and
        # top right corner
        # cropping the image
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[
            :, rmin:rmax, cmin:cmax]

        if desig[:8] == 'data_syn':
            # this might lead to problems later because we also use test data as background. But for now at first it is fine
            seed = random.choice(self._real)
            back = np.array(self._trancolor(Image.open(
                '{0}/{1}-color.png'.format(self._p_ycb, seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img

            try:
                background_img = Image.open(
                    '{0}/{1}-color.png'.format(self._p_ycb, desig))
            except:
                logging.info('cant find background')
        else:
            img_masked = img

        if self._cfg_d['noise_cfg']['status'] and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + \
                front[:, rmin:rmax, cmin:cmax] * \
                ~(mask_front[rmin:rmax, cmin:cmax])

        if desig[:8] == 'data_syn':
            img_masked = img_masked + \
                np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        # check how many pixels/points are within the masked area
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        # choose is a flattend array containg all pixles/points that are part of the object
        if len(choose) > self._num_pt:
            # randomly sample some points choose since object is to big
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self._num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            # take some padding around the tiny box
            choose = np.pad(choose, (0, self._num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten(
        )[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self._xmap[rmin:rmax, cmin:cmax].flatten(
        )[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self._ymap[rmin:rmax, cmin:cmax].flatten(
        )[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam[0]) * pt2 / cam[2]
        pt1 = (xmap_masked - cam[1]) * pt2 / cam[3]
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        cloud = np.add(cloud, add_t)

        dellist = [j for j in range(0, len(self._pcd_cad_dict[obj_idx]))]
        if self._cfg_d['output_cfg']['refine']:
            dellist = random.sample(dellist, len(
                self._pcd_cad_dict[obj_idx]) - self._num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(
                self._pcd_cad_dict[obj_idx]) - self._num_pt_mesh_small)
        model_points = np.delete(self._pcd_cad_dict[obj_idx], dellist, axis=0)

        # adds noise to target to regress on
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t + add_t)

        tup = (torch.from_numpy(cloud.astype(np.float32)),
               torch.LongTensor(choose.astype(np.int32)),
               self._norm(torch.from_numpy(img_masked.astype(np.float32))),
               torch.from_numpy(target.astype(np.float32)),
               torch.from_numpy(model_points.astype(np.float32)),
               torch.LongTensor([int(obj_idx) - 1]))

        if self._cfg_d['output_cfg']['ff']['status']:
            # ff is passed as tensor
            if self._cfg_d['output_cfg']['ff']['prediction'] == 'ground_truth':
                ff_rot = gt_rot_wxyz
                ff_trans = gt_trans

            elif self._cfg_d['output_cfg']['ff']['prediction'] == 'noise':
                np_rand_angles = np.random.normal(
                    0, self._cfg_d['output_cfg']['ff']['noise_degree'], (1, 3))
                r_random = R.from_euler(
                    'zyx', np_rand_angles, degrees=True)

                ff_rot = np.dot(gt_homo[:3, :3], r_random.as_matrix()[0, :, :])
                ff_trans = gt_trans + \
                    np.random.normal(
                        0, self._cfg_d['output_cfg']['ff']['noise_m'], (1, 3))
                ff_rot = re_quat(R.from_matrix(
                    ff_rot).as_quat(), 'xyzw')

            elif self._cfg_d['output_cfg']['ff']['prediction'] == 'time_lag':
                current_frame = int(desig.split('/')[-1])
                return_gt = False
                new_frame_idx = current_frame - \
                    self._cfg_d['output_cfg']['ff']['time_lag_frames']
                if new_frame_idx < 0:
                    return_gt = True
                else:
                    # assume past frame exists and ask for forgiveness if not
                    try:
                        new_desig = desig.split(
                            '/')[0] + f'/{new_frame_idx}'
                        meta2 = np.load(
                            f'{self._path}/processed/{new_desig}_meta.npy', allow_pickle=True)

                        homo_1 = copy.copy(meta2.item().get('pose_se3'))
                        homo_2 = np.eye(4)
                        homo_2[:3, :3] = r.as_matrix()
                        gt_homo_lag = np.dot(homo_2, homo_1)

                        ff_rot = re_quat(R.from_matrix(
                            gt_homo_lag[:3, :3]).as_quat(), 'xyzw').reshape((1, 4))
                        ff_trans = gt_homo_lag[:3, 3]
                    except:
                        return_gt = True
                if return_gt:
                    ff_rot = gt_rot
                    ff_trans = gt_trans

            ff = (torch.from_numpy(ff_trans.reshape((3)).astype(np.float32)),
                  torch.from_numpy(ff_rot.reshape((4)).astype(np.float32)))
            tup += ff

        else:
            tup += (0, 0)

        if self._cfg_d['output_cfg']['add_depth_image']:
            tup += tuple([np.transpose(depth[rmin:rmax, cmin:cmax], (1, 0))])
        else:
            tup += tuple([0])

        if self._cfg_d['output_cfg']['visu']['status']:
            # append visu information
            if self._cfg_d['output_cfg']['visu']['return_img']:
                info = (img_copy, cam)
            else:
                info = (0, cam)

            tup += (info)
        else:
            tup += (0, 0)

        gt_rot_wxyz = re_quat(
            R.from_matrix(target_r).as_quat(), 'xyzw')
        gt_trans = np.squeeze(target_t + add_t, 0)
        unique_desig = (desig, obj_idx)

        tup = tup + (gt_rot_wxyz, gt_trans, unique_desig)

        return tup

    def get_desig(self, path):
        desig = []
        with open(path) as f:
            for line in f:
                if line[-1:] == '\n':
                    desig.append(line[:-1])
                else:
                    desig.append(line)
        return desig

    def convert_desig_to_batch_list(self, desig, lookup_desig_to_obj):
        """ only works without sequence setting """

        if self._cfg_d['batch_list_cfg']['seq_length'] == 1:
            seq_list = []
            for d in desig:
                for o in lookup_desig_to_obj[d]:

                    obj_full_path = d[:-7]

                    obj_name = o
                    index_list = []
                    index_list.append(d.split('/')[-1])
                    seq_info = [obj_name, obj_full_path, index_list]
                    seq_list.append(seq_info)
        else:
            seq_added = 0
            # this method assumes that the desig list is sorted correctly
            # only adds synthetic data if present in desig list if fixed lendth = false

            seq_list = []
            # used frames keep max length to 10000 d+str(o) is the content
            used_frames = []
            mem_size = 10 * self._cfg_d['batch_list_cfg']['seq_length']
            total = len(desig)
            start = time.time()
            for j, d in enumerate(desig):
                print(f'progress: {j}/{total} time: {time.time()-start}')
                # limit memory for faster in search
                if len(used_frames) > mem_size:
                    used_frames = used_frames[-mem_size:]

                # tries to generate s sequence out of each object in the frame
                # memorize which frames we already added to a sequence
                for o in lookup_desig_to_obj[d]:

                    if not d + '_obj_' + str(o) in used_frames:

                        # try to run down the full sequence

                        if d.find('syn') != -1:
                            # synthetic data detected
                            if not self._cfg_d['batch_list_cfg']['fixed_length']:
                                # add the frame to seq_list
                                # object_name, full_path, index_list
                                seq_info = [o, d, [d.split('/')[-1]]]
                                seq_list.append(seq_info)
                                used_frames.append(d + '_obj_' + str(o))
                                # cant add synthetic data because not in sequences

                        else:
                            # no syn data
                            seq_idx = []
                            store = False
                            used_frames_tmp = []
                            used_frames_tmp.append(d + '_obj_' + str(o))

                            seq = int(d.split('/')[1])

                            seq_idx.append(int(desig[j].split('/')[-1]))
                            k = j
                            while len(seq_idx) < self._cfg_d['batch_list_cfg']['seq_length']:
                                k += self._cfg_d['batch_list_cfg']['sub_sample']
                                # check if same seq or object is not present anymore
                                if k < total:
                                    if seq != int(desig[k].split('/')[1]) or not (o in lookup_desig_to_obj[desig[k]]):
                                        if self._cfg_d['batch_list_cfg']['fixed_length']:
                                            store = False
                                            break
                                        else:
                                            store = True
                                            break
                                    else:
                                        seq_idx.append(
                                            int(desig[k].split('/')[-1]))
                                        used_frames_tmp.append(
                                            desig[k] + '_obj_' + str(o))
                                else:
                                    if self._cfg_d['batch_list_cfg']['fixed_length']:
                                        store = False
                                        break
                                    else:
                                        store = True
                                        break

                            if len(seq_idx) == self._cfg_d['batch_list_cfg']['seq_length']:
                                store = True

                            if store:

                                seq_info = [o, d[:-7], seq_idx]
                                seq_list.append(seq_info)
                                used_frames += used_frames_tmp
                                store = False
        return seq_list

    def get_batch_list(self):
        """create batch list based on cfg"""
        lookup_arr = np.load(
            self._cfg_env['p_ycb_lookup_desig_to_obj'], allow_pickle=True)

        lookup_dict = {}
        for i in range(lookup_arr.shape[0]):
            lookup_dict[lookup_arr[i, 0]] = lookup_arr[i, 1]

        if self._cfg_d['batch_list_cfg']['mode'] == 'dense_fusion_test':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_dense_test'])
            self._cfg_d['batch_list_cfg']['fixed_length'] = True
            self._cfg_d['batch_list_cfg']['seq_length'] = 1

        elif self._cfg_d['batch_list_cfg']['mode'] == 'dense_fusion_train':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_dense_train'])
            self._cfg_d['batch_list_cfg']['fixed_length'] = True
            self._cfg_d['batch_list_cfg']['seq_length'] = 1

        elif self._cfg_d['batch_list_cfg']['mode'] == 'train':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_seq_train'])

        elif self._cfg_d['batch_list_cfg']['mode'] == 'train_inc_syn':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_seq_train_inc_syn'])

        elif self._cfg_d['batch_list_cfg']['mode'] == 'test':
            desig_ls = self.get_desig(self._cfg_env['p_ycb_seq_test'])
        else:
            raise AssertionError

        # this is needed to add noise during runtime
        self._syn = self.get_desig(self._cfg_env['p_ycb_syn'])
        self._real = self.get_desig(self._cfg_env['p_ycb_seq_train'])
        name = str(self._cfg_d['batch_list_cfg'])
        name = name.replace("""'""", '')
        name = name.replace(" ", '')
        name = name.replace(",", '_')
        name = name.replace("{", '')
        name = name.replace("}", '')
        name = name.replace(":", '')
        name = self._cfg_env['p_ycb_config'] + '/' + name + '.pkl'
        try:
            with open(name, 'rb') as f:
                batch_ls = pickle.load(f)
        except:
            batch_ls = self.convert_desig_to_batch_list(desig_ls, lookup_dict)

            pickle.dump(batch_ls, open(name, "wb"))

        return batch_ls

    def get_camera(self, desig):
        """
        make this here simpler for cameras
        """
        if desig[:8] != 'data_syn' and int(desig[5:9]) >= 60:
            cx_2 = 323.7872
            cy_2 = 279.6921
            fx_2 = 1077.836
            fy_2 = 1078.189
            return np.array([cx_2, cy_2, fx_2, fy_2])
        else:
            cx_1 = 312.9869
            cy_1 = 241.3109
            fx_1 = 1066.778
            fy_1 = 1067.487
            return np.array([cx_1, cy_1, fx_1, fy_1])

    def get_pcd_cad_models(self):
        p = self._cfg_env['p_ycb_obj']
        class_file = open(p)
        cad_paths = []
        obj_idx = 1

        name_to_idx = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            if self._obj_list_fil is not None:
                for o in self._obj_list_fil:
                    if class_input.find(o) != -1:
                        cad_paths.append(
                            self._cfg_env['p_ycb'] + '/models/' + class_input[:-1])
                        name_to_idx[class_input[:-1]] = obj_idx
            else:
                cad_paths.append(
                    self._cfg_env['p_ycb'] + '/models/' + class_input[:-1])
                name_to_idx[class_input[:-1]] = obj_idx

            obj_idx += 1

        if len(cad_paths) == 0:
            raise AssertionError

        cad_dict = {}

        for path in cad_paths:
            input_file = open(
                '{0}/points.xyz'.format(path))

            cld = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                cld.append([float(input_line[0]), float(
                    input_line[1]), float(input_line[2])])
            cad_dict[name_to_idx[path.split('/')[-1]]] = np.array(cld)
            input_file.close()

        return cad_dict, name_to_idx

    @ property
    def visu(self):
        return self._cfg_d['output_cfg']['visu']['status']

    @ visu.setter
    def visu(self, vis):
        self._cfg_d['output_cfg']['visu']['status'] = vis

    @ property
    def refine(self):
        return self._cfg_d['output_cfg']['refine']

    @ refine.setter
    def refine(self, refine):
        self._cfg_d['output_cfg']['refine'] = refine
