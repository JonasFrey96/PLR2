from helper import re_quat
from torch.nn import functional as F
import cv2
import torchvision.transforms as transforms
import torch
import time
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy.ma as ma
import copy
import scipy.io as scio
import torch.utils.data as data
from PIL import Image, ImageOps
import logging
import os
import pickle
import glob
from loaders_v2 import Backend, ConfigLoader

_xmap = np.array([[j for i in range(320)] for j in range(240)])
_ymap = np.array([[i for i in range(320)] for j in range(240)])

def _read_keypoint_file(filepath):
    keypoints = []
    with open(filepath, 'rt') as f:
        while True:
            keypoint = np.zeros(3, dtype=np.float32)
            line = f.readline()
            if line == '':
                break
            vertex_id, x, y, z = line.split(' ')
            x, y, z = (float(i) for i in (x, y, z))
            keypoint[0] = x
            keypoint[1] = y
            keypoint[2] = z
            keypoints.append(keypoint)
    return np.stack(keypoints)

class ImageExtractor:
    def __init__(self, desig, img, depth, label, meta, object_ids, num_points,
                 pcd_to_cad, keypoints=None):
        self.desig = desig
        self.depth = depth
        self.label = label
        self.meta = meta
        self.object_ids = object_ids
        self._pcd_to_cad = pcd_to_cad
        self.cam = self.get_camera(desig)
        self.keypoints = keypoints
        self._extract_poses()

        self.rmin = 0
        self.rmax = 480
        self.cmin = 0
        self.cmax = 640
        self._choose = np.full((480, 640), True).flatten()
        self.img = img
        self.pcd = self.pointcloud_layered()

    def _extract_poses(self):
        N = self.meta['cls_indexes'].size
        self.R = np.zeros((N, 3, 3))
        self.t = np.zeros((N, 1, 3))
        for i in self.object_ids:
            obj_idx_in_list = int(np.argwhere(self.object_ids == i))
            self.R[obj_idx_in_list] = self.meta['poses'][:, :, obj_idx_in_list][:, 0:3]
            self.t[obj_idx_in_list, 0, :] = np.array([self.meta['poses'][:, :, obj_idx_in_list][:, 3:4].flatten()])

    def _get_pose(self, object_id):
        index_in_list = int(np.argwhere(self.object_ids == object_id))
        return self.R[index_in_list], self.t[index_in_list]

    def pointcloud_layered(self):
        cam_scale = self.meta['factor_depth'][0][0]
        cam_cx, cam_cy, cam_fx, cam_fy = self.cam / 2.0
        pt2 = self.depth / cam_scale
        pt0 = (_ymap - cam_cx) * pt2 / cam_fx
        pt1 = (_xmap - cam_cy) * pt2 / cam_fy
        return np.dstack((pt0, pt1, pt2)).astype(np.float32)

    def get_camera(self, desig):
        """
        make this here simpler for cameras
        """
        intrinsics = self.meta['intrinsic_matrix']
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        return np.array([cx, cy, fx, fy])

    def model_points(self, refine, points_small, points_large):
        dellist = [j for j in range(0, len(self._pcd_to_cad[self.obj_idx]))]
        if refine:
            dellist = random.sample(dellist, len(
                self._pcd_to_cad[self.obj_idx]) - points_large)
        else:
            dellist = random.sample(dellist, len(
                self._pcd_to_cad[self.obj_idx]) - points_small)
        model_points = np.delete(
            self._pcd_to_cad[self.obj_idx], dellist, axis=0)
        return model_points

    def keypoint_vectors(self):
        H, W, _ = self.pcd.shape
        K = self.keypoints[1].shape[0]
        vectors = np.zeros((H, W, K, 3), dtype=np.float32)

        for object_id in self.object_ids:
            object_mask = self.label == object_id
            # mask_obj = np.dstack([(self.label==object_id)] * 3)
            points = self.pcd[object_mask, :]
            R, t = self._get_pose(object_id)
            keypoints = self.keypoints[object_id]
            keypoints = (R @ keypoints[:, :, None])[:, :, 0] + t

            vectors[object_mask, :, :] = keypoints[None] - points[:, None]
        return vectors.reshape(H, W, K*3)

    def center_vectors(self):
        H, W = self.label.shape
        out = np.zeros((H, W, 3), dtype=np.float32)
        for object_id in self.object_ids:
            object_mask = self.label == object_id
            _, t = self._get_pose(object_id)
            out[object_mask, :] = t[0, :] - self.pcd[object_mask, :]
        return out

LEFT_RIGHT_FLIP = np.array([[-1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]], dtype=np.float32)

def flip_meta(meta, width):
    poses = meta['poses']
    for i in range(poses.shape[2]):
        t = poses[:, 3:4, i]
        R = poses[:, :3, i]
        poses[:, :3, i] = LEFT_RIGHT_FLIP @ R
        poses[:, 3:4, i] = LEFT_RIGHT_FLIP @ t
    meta['poses'] = poses

    center_x = width / 2.0
    cx = meta['intrinsic_matrix'][0, 2]

    diff = (center_x - cx)
    cx = center_x + diff
    meta['intrinsic_matrix'][0, 2] = cx

    return meta

class YCB(Backend):
    def __init__(self, cfg_d, cfg_env):
        super(YCB, self).__init__(cfg_d, cfg_env)
        self._dataset_config = cfg_d
        self._env_config = cfg_env
        self._ycb_path = cfg_env['p_ycb']
        self._pcd_cad_dict, self._name_to_idx, self._keypoints = self._read_model_files()
        self._object_count = len(self._pcd_cad_dict)
        self._batch_list = self.get_batch_list()

        self._length = len(self._batch_list)

    @property
    def object_models(self):
        return self._pcd_cad_dict

    @property
    def keypoints(self):
        return self._keypoints

    def getFullImage(self, desig):
        """
        Desig: sequence/idx
        Gets the full image.
        """
        try:
            img = Image.open(
                '{0}/{1}-color.png'.format(self._ycb_path, desig)).convert("RGB")
            depth = Image.open(
                '{0}/{1}-depth.png'.format(self._ycb_path, desig))
            label = Image.open(
                '{0}/{1}-label.png'.format(self._ycb_path, desig))
            meta = scio.loadmat(
                '{0}/{1}-meta.mat'.format(self._ycb_path, desig))

        except FileNotFoundError:
            logging.error(
                'cant find files for {0}/{1}'.format(self._ycb_path, desig))
            return False

        p_flip = self._dataset_config.get('p_flip', 0.0)
        if np.random.uniform(0, 1) < p_flip:
            img = ImageOps.mirror(img)
            depth = ImageOps.mirror(depth)
            label = ImageOps.mirror(label)
            meta = flip_meta(meta, img.size[0])

        depth = np.array(depth.resize((320, 240), Image.BILINEAR))
        label = np.array(label.resize((320, 240), Image.NEAREST), dtype=np.int32)

        object_ids = meta['cls_indexes'].flatten().astype(np.int)

        extractor = ImageExtractor(desig, img, depth, label, meta, object_ids,
                self._num_pt, self._pcd_cad_dict, self._keypoints)

        cloud = extractor.pcd
        cam = extractor.cam

        keypoint_vectors = extractor.keypoint_vectors()
        center_vectors = extractor.center_vectors()

        cloud = cloud.transpose([2, 0, 1]) # H x W x C -> C x H x W
        img = (np.array(img).astype(np.float32) / 127.5 - 1.0).transpose([2, 0, 1])
        keypoint_vectors = torch.from_numpy(keypoint_vectors.transpose([2, 0, 1]))
        center_vectors = torch.from_numpy(center_vectors.transpose([2, 0, 1]))
        label = torch.from_numpy(label)

        tup = (torch.from_numpy(cloud.astype(np.float32)),
               torch.from_numpy(img),
               label.to(torch.long),
               keypoint_vectors,
               center_vectors)

        if self._dataset_config['output_cfg']['visu']['status']:
            # append visu information
            tup += (cam,)
        else:
            tup += (0,)

        objects_in_scene = np.zeros((self._object_count,))
        objects_in_scene[object_ids - 1] = 1.0
        return tup + (objects_in_scene, desig)

    def add_linear_motion_blur(self, img):
        ## Code adapted from He. at al (github user ethnhe):
        ## authors of "PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation, CVPR 2020."
        ## code available at https://github.com/ethnhe/PVN3D
        ## MIT licence applies.
        r_angle = np.deg2rad(np.random.randint(0, self._dataset_cfg))
        length = np.random.rand()*15+1

        dx = np.cos(r_angle)
        dy = np.sin(r_angle)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def get_normal(self, cld):
        """Taken from He. at al (github user ethnhe):
        Authors of "PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation, CVPR 2020."
        Code available at https://github.com/ethnhe/PVN3D
        MIT licence applies."""
        cloud = pcl.PointCloud()
        cld = cld.astype(np.float32)
        cloud.from_array(cld)
        ne = cloud.make_NormalEstimation()
        kdtree = cloud.make_kdtree()
        ne.set_SearchMethod(kdtree)
        ne.set_KSearch(50)
        n = ne.compute()
        n = n.to_array()
        return n

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

        if self._dataset_config['batch_list_cfg']['seq_length'] == 1:
            seq_list = []
            for d in desig:
                object_list = lookup_desig_to_obj.get(d, [])
                for o in object_list:

                    obj_full_path = d[:-7]

                    obj_name = o
                    index_list = []
                    index_list.append(d.split('/')[-1])
                    seq_info = [obj_name, obj_full_path, index_list]
                    seq_list.append(seq_info)
        else:
            # this method assumes that the desig list is sorted correctly
            # only adds synthetic data if present in desig list if fixed length = false

            seq_list = []
            # used frames keep max length to 10000 d+str(o) is the content
            used_frames = []
            mem_size = 10 * \
                self._dataset_config['batch_list_cfg']['seq_length']
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
                            if not self._dataset_config['batch_list_cfg']['fixed_length']:
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
                            while len(seq_idx) < self._dataset_config['batch_list_cfg']['seq_length']:
                                k += self._dataset_config['batch_list_cfg']['sub_sample']
                                # check if same seq or object is not present anymore
                                if k < total:
                                    if seq != int(desig[k].split('/')[1]) or not (o in lookup_desig_to_obj[desig[k]]):
                                        if self._dataset_config['batch_list_cfg']['fixed_length']:
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
                                    if self._dataset_config['batch_list_cfg']['fixed_length']:
                                        store = False
                                        break
                                    else:
                                        store = True
                                        break

                            if len(seq_idx) == self._dataset_config['batch_list_cfg']['seq_length']:
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
            self._env_config['p_ycb_lookup_desig_to_obj'], allow_pickle=True)

        lookup_dict = {}
        for i in range(lookup_arr.shape[0]):
            lookup_dict[lookup_arr[i, 0]] = lookup_arr[i, 1]

        if self._dataset_config['batch_list_cfg']['mode'] == 'dense_fusion_test':
            desig_ls = self.get_desig(self._env_config['p_ycb_dense_test'])
            self._dataset_config['batch_list_cfg']['fixed_length'] = True
            self._dataset_config['batch_list_cfg']['seq_length'] = 1

        elif self._dataset_config['batch_list_cfg']['mode'] == 'dense_fusion_train':
            desig_ls = self.get_desig(self._env_config['p_ycb_dense_train'])
            self._dataset_config['batch_list_cfg']['fixed_length'] = True
            self._dataset_config['batch_list_cfg']['seq_length'] = 1

        elif self._dataset_config['batch_list_cfg']['mode'] == 'train':
            desig_ls = self.get_desig(self._env_config['p_ycb_seq_train'])

        elif self._dataset_config['batch_list_cfg']['mode'] == 'syn':
            desig_ls = _list_synthetic_data(self._ycb_path)
        elif self._dataset_config['batch_list_cfg']['mode'] == 'train_inc_syn':
            desig_ls = self.get_desig(
                self._env_config['p_ycb_seq_train_inc_syn'])

        elif self._dataset_config['batch_list_cfg']['mode'] == 'test':
            desig_ls = self.get_desig(self._env_config['p_ycb_seq_test'])
        else:
            raise AssertionError

        # this is needed to add noise during runtime
        # name = str(self._dataset_config['batch_list_cfg'])
        # name = name.replace("""'""", '')
        # name = name.replace(" ", '')
        # name = name.replace(",", '_')
        # name = name.replace("{", '')
        # name = name.replace("}", '')
        # name = name.replace(":", '')
        # name = self._env_config['p_ycb_config'] + '/' + name + '.pkl'
        # try:
        #     with open(name, 'rb') as f:
        #         batch_ls = pickle.load(f)
        # except FileNotFoundError:
        batch_ls = self.convert_desig_to_batch_list(desig_ls, lookup_dict)

        # pickle.dump(batch_ls, open(name, "wb"))

        return batch_ls

    def _read_model_files(self):
        p = self._env_config['p_ycb_obj']
        with open(p, 'rt') as class_file:
            cad_dict, name_to_idx  = self._build_pcd_cad_dict(class_file)

        keypoints = self._load_keypoints(name_to_idx)

        return cad_dict, name_to_idx, keypoints

    def _build_pcd_cad_dict(self, class_file):
        cad_paths = []
        obj_idx = 1

        name_to_idx = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            if self._obj_list_fil is not None:
                if obj_idx in self._obj_list_fil:
                  cad_paths.append(
                      self._env_config['p_ycb'] + '/models/' + class_input[:-1])
                  name_to_idx[class_input[:-1]] = obj_idx
            else:
                cad_paths.append(
                    self._env_config['p_ycb'] + '/models/' + class_input[:-1])
                name_to_idx[class_input[:-1]] = obj_idx

            obj_idx += 1

        if len(cad_paths) == 0:
            raise AssertionError

        cad_dict = {}

        for path in cad_paths:
            input_file = open(os.path.join(path, 'points.xyz'))

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

    def _load_keypoints(self, name_to_idx):
        keypoints = {}
        for name, object_index in name_to_idx.items():
            keypoint_path = os.path.join(self._ycb_path, 'models', name, "textured.landmarks")
            keypoints[object_index] = _read_keypoint_file(keypoint_path)
        return keypoints

    @ property
    def visu(self):
        return self._dataset_config['output_cfg']['visu']['status']

    @ visu.setter
    def visu(self, vis):
        self._dataset_config['output_cfg']['visu']['status'] = vis

    @ property
    def refine(self):
        return self._dataset_config['output_cfg']['refine']

    @ refine.setter
    def refine(self, refine):
        self._dataset_config['output_cfg']['refine'] = refine

def _list_synthetic_data(ycb_path):
    synthetic_data = os.path.join(ycb_path, 'data_syn', '*-color.png')
    synthetic = []
    paths = glob.iglob(synthetic_data)
    for path in paths:
        desig = path.replace(ycb_path, '').replace('-color.png', '')[1:]
        synthetic.append(desig)
    return synthetic[:5]

