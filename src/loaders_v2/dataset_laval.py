import os
import sys
from PIL import Image
import numpy as np
import random
import time
import logging
import coloredlogs
import copy
import torch
import torch.utils.data as data
import random
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

from estimation.state import State_R3xQuat, State_SE3, points
from helper import compose_quat, rotation_angle, re_quat
from visu import plot_pcd, plot_two_pcd
from helper import generate_unique_idx
from loaders_v2 import Backend, ConfigLoader


class Laval(Backend):
    def __init__(self, cfg_d, cfg_env):
        super(Laval, self).__init__(cfg_d, cfg_env)
        self._pcd_cad_dict, self._name_to_idx = self.get_pcd_cad_models(
            cfg_env['p_laval'])
        self._batch_list = self.get_batch_list(
            cfg_env['p_laval'], cfg_d['batch_list_cfg'])

        # try statement to be compatible with older exp configs
        try:
            if cfg_d['batch_list_cfg']['add_syn_to_train']:
                inp = cfg_d['batch_list_cfg']
                inp['mode'] = 'syn'
                self._batch_list_syn = self.get_batch_list_syn(
                    cfg_env['p_laval'] + '_syn', inp)

                self._batch_list_real = copy.deepcopy(self._batch_list)

                self._batch_list += self._batch_list_syn
        except:
            pass

        # filter batch_list and pcd_cad_dict according to obj_list
        self._length = len(self._batch_list)
        self._noise_cfg = cfg_d['noise_cfg']
        self._output_cfg = cfg_d['output_cfg']
        self._path = cfg_env['p_laval']
        self._camera_dict = self.get_camera_dict(cfg_env['p_laval'])

        self._norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)  # 0.05

    def getModelPoints(self, desig, obj):
        idx = generate_unique_idx(
            self._pcd_cad_dict[obj].shape[0], self._pcd_cad_dict[obj].shape[0])
        meta = np.load(
            f'{self._path}/processed/{desig}_meta.npy', allow_pickle=True)
        return self._pcd_cad_dict[obj][idx], meta.item().get('pose_se3')

    def getElement(self, desig, obj):
        """
        desig : sequence/idx
        two problems we face. What is if an object is not visible at all -> meta['obj'] = None
        what if to few points are on the object (i would suggest sample points mutiple times)
        """
        self._pcd_cad_dict[obj]
        # plot_pcd(self._pcd_cad_dict[obj])
        try:
            if desig.find('processed/') != -1:
                meta = np.load(
                    f'{self._path}/{desig}_meta.npy', allow_pickle=True)
                pcd_target = np.load(
                    f'{self._path}/{desig}_target.npy', allow_pickle=True)
                choose = np.load(
                    f'{self._path}/{desig}_choose.npy', allow_pickle=True)
                img_crop = np.array(self._trancolor(Image.open(
                    f'{self._path}/{desig}_img_crop.png').convert("RGB")))
                syn = False

            elif desig.find('_syn/') != -1:
                # access normal data
                meta = np.load(
                    f'{self._path}{desig}_meta.npy', allow_pickle=True)

                img_crop = np.array(Image.open(
                    f'{self._path}{desig}_img_crop.png'))

                # access background image by random sampling
                s_rnd = np.random.randint(0, len(self._batch_list_real))
                f_rnd = np.random.randint(
                    0, len(self._batch_list_real[s_rnd][2]))
                # print(f'{self._path}/{desig}/{f_rnd}.png')
                img_background = np.array(Image.open(
                    f'{self._path}/{self._batch_list_real[s_rnd][1]}/{f_rnd}.png'))

                # add the background to the croped syn image
                h = np.random.randint(
                    0, img_background.shape[0] - img_crop.shape[0])
                w = np.random.randint(
                    0, img_background.shape[1] - img_crop.shape[1])
                _hs, _ws = np.where(img_crop[:, :, 3] == 0)
                _hs2 = _hs + img_crop.shape[0]
                _ws2 = _ws + img_crop.shape[1]
                i0 = Image.fromarray(img_crop[:, :, :3])
                img_crop[_hs, _ws, :3] = img_background[_hs2, _ws2, :3]
                tmp = img_crop[:, :, :3]
                img_crop = np.array(
                    self._trancolor(Image.fromarray(tmp).convert("RGB")))

                syn = True

                pcd_target = np.load(
                    f'{self._path}{desig}_target.npy', allow_pickle=True)
                choose = np.load(
                    f'{self._path}{desig}_choose.npy', allow_pickle=True)

            else:
                raise Exception
        except:
            print(f'CANT ACCESS DATA WITH DESIGNATOR {desig}')
        if choose.shape[0] < self._num_pt:
            pass
            # print("ERROR not enough points")

        num_pt = self._num_pt
        idx = generate_unique_idx(
            num_pt, pcd_target.shape[0])
        choose = choose[idx]
        cloud = pcd_target[idx]

        # sample size of pointcloud
        if self._output_cfg['refine']:
            num_pt = self._num_pt_mesh_large
        else:
            num_pt = self._num_pt_mesh_small

        idx = generate_unique_idx(
            num_pt, self._pcd_cad_dict[obj].shape[0])

        model_points = self._pcd_cad_dict[obj][idx]

        homo = meta.item().get('pose_se3')
        target = np.dot(model_points, homo[:3, :3].T)
        target = np.add(target, homo[:3, 3])

        if syn:
            gt_homo = copy.copy(homo)
        else:
            # rotation post trans since this is stored in the wrong way in the meta data !
            r = R.from_euler('x', 180, degrees=True)
            target = np.dot(target, r.as_matrix())

            homo_1 = copy.copy(homo)
            homo_2 = np.eye(4)
            homo_2[:3, :3] = r.as_matrix()
            gt_homo = np.dot(homo_2, homo_1)

        gt_rot = re_quat(R.from_matrix(gt_homo[:3, :3]).as_quat(), 'xyzw')
        gt_trans = gt_homo[:3, 3]

        bounds = meta.item().get('b')
        if self._noise_cfg['status']:
            pass

        img_crop = np.transpose(img_crop, (2, 0, 1))

        keys = ['points', 'choose', 'img', 'target', 'model_points', 'idx', 'ff_trans',
                'ff_rot', 'img_org', 'cam_cal', 'gt_rot_wxyz', 'gt_trans']
        # TODO object_name to index mapping
        test = torch.LongTensor([int(self._name_to_idx[obj])])
        if isinstance(test, list):
            test = torch.LongTensor([int(0)])
            print("ERRRPRRRR")

        # passed by tensor
        tup = (torch.from_numpy(cloud.astype(np.float32)),
               torch.LongTensor(choose.T.astype(np.int32)),
               self._norm(torch.from_numpy(img_crop.astype(np.float32))),
               torch.from_numpy(target.astype(np.float32)),
               torch.from_numpy(model_points.astype(np.float32)),
               test)

        if self._output_cfg['ff']['status']:
            # ff is passed as tensor
            if self._output_cfg['ff']['prediction'] == 'ground_truth':
                ff_rot = gt_rot
                ff_trans = gt_trans
            elif self._output_cfg['ff']['prediction'] == 'noise':
                np_rand_angles = np.random.normal(
                    0, self._output_cfg['ff']['noise_degree'], (1, 3))
                r_random = R.from_euler(
                    'zyx', np_rand_angles, degrees=True)

                ff_rot = np.dot(gt_homo[:3, :3], r_random.as_matrix()[0, :, :])
                ff_trans = gt_trans + \
                    np.random.normal(
                        0, self._output_cfg['ff']['noise_m'], (1, 3))

                ff_rot = re_quat(R.from_matrix(
                    ff_rot).as_quat(), 'xyzw')

            elif self._output_cfg['ff']['prediction'] == 'time_lag':
                current_frame = int(desig.split('/')[-1])
                return_gt = False
                new_frame_idx = current_frame - \
                    self._output_cfg['ff']['time_lag_frames']
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
            a = np.zeros((1))
            b = np.zeros((1))
            tup += (torch.from_numpy(a.astype(np.float32)),
                    torch.from_numpy(b.astype(np.float32)))

        if (self._output_cfg['visu']['status'] == True):
            # everything that is needed for visu is passed as numpy array
            if syn:
                cam_cal = np.asarray([meta.item().get('cx'), meta.item().get(
                    'cy'), meta.item().get('fx'), meta.item().get('fy')])
                if self._output_cfg['visu']['return_img']:
                    img_orig = np.array(Image.open(
                        f'{self._path}{desig}.png'))[:, :, :3]
                    cam = (img_orig, cam_cal)
                else:
                    cam = (np.zeros((1)), cam_cal)
            else:
                seq_name = desig.split('/')[-2]
                self._camera_dict[seq_name]
                cam_cx = self._camera_dict[seq_name].center_x
                cam_cy = self._camera_dict[seq_name].center_y
                cam_fx = self._camera_dict[seq_name].focal_x
                cam_fy = self._camera_dict[seq_name].focal_y
                cam_cal = np.asarray([cam_cx, cam_cy, cam_fx, cam_fy])

                if self._output_cfg['visu']['return_img']:
                    img_orig = np.array(Image.open(
                        f'{self._path}/{desig}.png'))

                    cam = (img_orig, cam_cal)
                else:
                    cam = (np.zeros((1)), cam_cal)
            tup += cam
        else:
            a = np.zeros((1))
            b = np.zeros((1))
            tup += (torch.from_numpy(a.astype(np.float32)),
                    torch.from_numpy(b.astype(np.float32)))

        # passed as numpy array
        tup += (gt_rot, gt_trans)
        unique_desig = tuple([desig])
        tup += unique_desig
        # check if an obj is available in this sequence
        if (meta.item().get('obj') is None and not syn) or pcd_target.shape[0] < 100:
            print('lenght of pcd_target:', pcd_target.shape[0])
            print("OBJ", meta.item().get('obj'))
            # if no obj is availalbe return none -> this is caught when loading the model later
            # therfore only the motion model is used for future timesteps
            return (False, target, model_points, torch.LongTensor([int(self._name_to_idx[obj])]), gt_rot, gt_trans, unique_desig)
        return tup

    def get_batch_list(self, path, cfg):
        name = 'seq_length-%d_fixed-%d_sub-%d' % (
            cfg['seq_length'], cfg['fixed_length'], cfg['sub_sample'])
        try:
            # try to load and ask for forgivness TODO
            print("fix loading back without xyz")
            ret = np.load(f"{path}/{name}xyz.npy", allow_pickle=True).tolist()
            return ret
        except:
            # seq_paths = [x[0] for x in os.walk(path)][1:]
            seq_paths = [x[0]
                         for x in os.walk(path) if (x[0].find('processed') != -1)][1:]
            # TODO
            if cfg['sequence_names'] is not None:
                for name in cfg['sequence_names']:
                    seq_paths = [x for x in seq_paths if x.find(name) != -1]

            if cfg['mode'] == 'fair_train':
                seq_paths_tmp = []
                for i, s in enumerate(seq_paths):
                    if i % 10 == 0:
                        pass
                    else:
                        seq_paths_tmp.append(s)
                seq_paths = seq_paths_tmp

            if cfg['mode'] == 'fair_test':
                seq_paths = seq_paths[::10]

            seq_list = []
            for seq in seq_paths:
                index_list = []
                i = 0
                while 1:
                    if os.path.exists(seq + '/%d_meta.npy' % (i)):
                        index_list.append(i)

                        if (i + 1) % int(cfg['seq_length'] * cfg['sub_sample']) == 0:
                            obj_name = seq.split('/')[-1].split('_')[0]
                            obj_full_path = 'processed/' + seq.split('/')[-1]

                            seq_info = [obj_name, obj_full_path, index_list]
                            seq_list.append(seq_info)
                            index_list = []
                    else:
                        if cfg['fixed_length'] == True or len(index_list) != -1:
                            break
                        else:
                            if len(seq_list) != 0:
                                obj_name = seq.split('/')[-1].split('_')[0]
                                obj_full_path = seq.split('/')[-1]
                                seq_info = [
                                    obj_name, obj_full_path, index_list]
                                seq_list.append(seq_info)
                            break
                    i += cfg['sub_sample']

            if cfg['mode'] == 'train':
                idx_train = [x for x in range(
                    0, len(seq_list)) if x % 10 != 0]

                batch_list_np = np.array(seq_list)[idx_train]
            elif cfg['mode'] == 'test':
                seq_list = seq_list[::10]
                batch_list_np = np.array(seq_list)
            elif cfg['mode'] == 'visu' or cfg['mode'] == 'fair_test' or cfg['mode'] == 'fair_train':
                batch_list_np = np.array(seq_list)
            else:
                raise Exception

            np.save(f"{path}/{name}.npy", batch_list_np)
            return batch_list_np.tolist()

    def get_batch_list_syn(self, path, cfg):
        seq_paths = []
        for x in os.walk(path):
            try:
                if (int(x[0].split('_')[-1]) < 74):
                    seq_paths.append(x[0])
            except:
                pass
        # print(seq_paths)
        # seq_paths = [x[0]
        #              for x in os.walk(path) if (int(x[0].split('/')[-1]) < 74)]  # x[0].find('laval_syn/') != -1

        if cfg['sequence_names'] is not None:
            for name in cfg['sequence_names']:
                seq_paths = [x for x in seq_paths if x.find(name) != -1]

        seq_list = []
        for seq in seq_paths:
            index_list = []
            i = 0
            while 1:
                if os.path.exists(seq + '/%d_meta.npy' % (i)):
                    index_list.append(i)

                    if (i + 1) % int(cfg['seq_length'] * cfg['sub_sample']) == 0:
                        obj_name = seq.split('/')[-1].split('_')[0]
                        obj_full_path = '_syn/' + seq.split('/')[-1]

                        seq_info = [obj_name, obj_full_path, index_list]
                        seq_list.append(seq_info)
                        index_list = []
                else:
                    if cfg['fixed_length'] == True or len(index_list) != -1:
                        break
                    else:
                        if len(seq_list) != 0:
                            obj_name = seq.split('/')[-1].split('_')[0]
                            obj_full_path = '_syn/' + seq.split('/')[-1]
                            seq_info = [
                                obj_name, obj_full_path, index_list]
                            seq_list.append(seq_info)
                        break
                i += cfg['sub_sample']

        if cfg['mode'] == 'syn':
            batch_list_np = np.array(seq_list)
        else:
            raise Exception

        return batch_list_np.tolist()

    def get_camera_dict(self, p):
        if self._obj_list_fil is None:
            obj = os.walk(p)
            seq_paths = [x[0]
                         for x in os.walk(p) if (x[0].find('processed') != -1)][1:]
        else:
            seq_paths = []
            for x in os.walk(p):
                if x[0].find('processed') != -1 and x[0].split('/')[-1].split('_')[0] in self._obj_list_fil:
                    seq_paths.append(x[0])

        camera_dict = {}
        for path in seq_paths:
            camera_dict[path.split(
                '/')[-1]] = Camera.load_from_json(path + '/')
        return camera_dict

    def get_pcd_cad_models(self, p):
        if self._obj_list_fil is None:
            cad_paths = [x[0]
                         for x in os.walk(p) if (x[0].find('processed') == -1)][1:]
        else:
            cad_paths = []
            for o in self._obj_list_fil:
                cad_paths.append(f'{p}/{o}')

        name_to_idx = {}
        cad_dict = {}
        idx = 0
        for path in cad_paths:
            pcd_tmp = np.asarray(
                o3d.io.read_point_cloud(path + '/geometry.ply').points)
            cad_dict[path.split('/')[-1]] = pcd_tmp
            name_to_idx[path.split('/')[-1]] = int(idx)
            idx += 1

        return cad_dict, name_to_idx

    @ property
    def visu(self):
        return self._output_cfg['visu']['status']

    @ visu.setter
    def visu(self, vis):
        self._output_cfg['visu']['status'] = vis

    @ property
    def refine(self):
        return self._output_cfg['refine']

    @ refine.setter
    def refine(self, refine):
        self._output_cfg['refine'] = refine
