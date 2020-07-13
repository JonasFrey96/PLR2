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
from PIL import Image
import logging
import os
import pickle
from loaders_v2 import Backend, ConfigLoader
from helper import get_bbox_480_640

_xmap = np.array([[j for i in range(640)] for j in range(480)])
_ymap = np.array([[i for i in range(640)] for j in range(480)])

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
    def __init__(self, desig, obj_idx, img, depth, label, meta, object_ids, num_points,
                 pcd_to_cad, keypoints=None):
        self.desig = desig
        self.obj_idx = obj_idx
        self.depth = depth
        self.label = label
        self.meta = meta
        self.object_ids = object_ids
        self._pcd_to_cad = pcd_to_cad
        self.cam = self.get_camera(desig)
        self.keypoints = keypoints

        if obj_idx is not None:
            # when less then this number of points are visble in the scene the frame is thrown away, not used in the overall image mask calculation
            self._num_pt = num_points
            self._minimum_num_pt = 50
            self.valid = self._compute_mask()
            if self.valid:
                self._compute_choose()
                self.img = self._crop_image(img)
        else:
            self.rmin = 0
            self.rmax = 480
            self.cmin = 0
            self.cmax = 640
            self._choose = np.full((480, 640), True).flatten()
            self.img = img
            self.pcd = self.pointcloud_layered()

    def _crop_image(self, img):
        # cropping the image
        return np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[
            :, self.rmin:self.rmax, self.cmin:self.cmax]

    def _compute_pose(self, obj_idx):
        obj = self.meta['cls_indexes'].flatten().astype(np.int32)

        obj_idx_in_list = int(np.argwhere(obj == obj_idx))
        R = self.meta['poses'][:, :, obj_idx_in_list][:, 0:3]
        t = np.array([self.meta['poses'][:, :, obj_idx_in_list][:, 3:4].flatten()])
        return R, t

    def _compute_mask(self):
        mask_depth = ma.getmaskarray(ma.masked_not_equal(self.depth, 0))
        if self.obj_idx is not None:
            mask_label = ma.getmaskarray(ma.masked_equal(self.label, self.obj_idx))
            self._mask = mask_label * mask_depth
        else:
            self._mask = mask_depth

        if len(self._mask.nonzero()[0]) <= self._minimum_num_pt:
            return False

        self.rmin, self.rmax, self.cmin, self.cmax = get_bbox_480_640(
            mask_label)

        return True

    def _compute_choose(self):
        # check how many pixels/points are within the masked area
        choose = self._mask[self.rmin:self.rmax, self.cmin:self.cmax].flatten().nonzero()[0]
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
        self._choose = choose

    def pointcloud(self):
        depth_masked = (self.depth[self.rmin:self.rmax, self.cmin:self.cmax]
                        .flatten()[self._choose][:, np.newaxis].astype(np.float32))
        xmap_masked = (_xmap[self.rmin:self.rmax, self.cmin:self.cmax]
                       .flatten()[self._choose][:, np.newaxis].astype(np.float32))
        ymap_masked = (_ymap[self.rmin:self.rmax, self.cmin:self.cmax]
                       .flatten()[self._choose][:, np.newaxis].astype(np.float32))

        cam_scale = self.meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam[0]) * pt2 / self.cam[2]
        pt1 = (xmap_masked - self.cam[1]) * pt2 / self.cam[3]
        return np.concatenate((pt0, pt1, pt2), axis=1)

    def pointcloud_layered(self):
        cam_scale = self.meta['factor_depth'][0][0]
        pt2 = self.depth / cam_scale
        pt0 = (_ymap - self.cam[0]) * pt2 / self.cam[2]
        pt1 = (_xmap - self.cam[1]) * pt2 / self.cam[3]
        return np.dstack((pt0, pt1, pt2)).astype(np.float32)

    def choose(self):
        return np.array([self._choose])

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

    def image_masked(self):
        mask_back = ma.getmaskarray(ma.masked_equal(self.label, 0))
        if self.desig[:8] == 'data_syn':
            # this might lead to problems later because we also use test data as background. But for now at first it is fine
            seed = random.choice(self._real)
            back = np.array(self._trancolor(self.img.convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[
                :, self.rmin:self.rmax, self.cmin:self.cmax]
            img_masked = back * \
                mask_back[self.rmin:self.rmax, self.cmin:self.cmax] + self.img
        else:
            # TODO: figure out if img_masked is supposed to be with the background masked out.
            img_masked = self.img
        return img_masked

    def mask(self):
        return self._mask

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
        num_keypoints = self.keypoints[1].shape[0]
        kv = [self.pcd.copy() for _ in range(8)]

        for obj in self.object_ids:
            mask_obj = np.dstack([(self.label==obj)] * 3)
            R, t = self._compute_pose(obj)
            kp = self.keypoints[obj]
            kp = (R @ kp[:, :, None])[:, :, 0] + t
            for i in range(0, num_keypoints):
                kp_vec = kp[i, :]
                kp_x = kp_vec[0] - kv[i][:, :, 0]
                kp_y = kp_vec[1] - kv[i][:, :, 1]
                kp_z = kp_vec[2] - kv[i][:, :, 2]
                keypoint_cloud = np.dstack((kp_x, kp_y, kp_z))
                kv[i][mask_obj] = keypoint_cloud[mask_obj]
        mask_back = np.dstack([(self.label==0)]*3)
        for i in range(0, num_keypoints):
            kv[i][mask_back] = 0
        return np.concatenate(kv, axis=2)

    def center_vectors(self):
        H, W = self.label.shape
        out = np.zeros((H, W, 3), dtype=np.float32)
        for object_id in self.object_ids:
            object_mask = self.label == object_id
            _, t = self._compute_pose(object_id)
            out[object_mask, :] = t[0, :] - self.pcd[object_mask, :]
        return out

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

        self._trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self._front_num = 2

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
                '{0}/{1}-color.png'.format(self._ycb_path, desig))
            depth = np.array(Image.open(
                '{0}/{1}-depth.png'.format(self._ycb_path, desig)))
            label = np.array(Image.open(
                '{0}/{1}-label.png'.format(self._ycb_path, desig))).astype(np.int32)
            meta = scio.loadmat(
                '{0}/{1}-meta.mat'.format(self._ycb_path, desig))

        except FileNotFoundError:
            logging.error(
                'cant find files for {0}/{1}'.format(self._ycb_path, desig))
            return False

        if self._dataset_config['output_cfg']['visu']['return_img']:
            img_copy = np.array(img.convert("RGB"))

        object_ids = np.unique(label)[1:]

        # take the noise color image
        if self._dataset_config['noise_cfg']['status']:
            img = self._trancolor(img)


        extractor = ImageExtractor(desig, None, img, depth, label, meta, object_ids,
                self._num_pt, self._pcd_cad_dict, self._keypoints)

        cloud = extractor.pcd
        cam = extractor.cam

        keypoint_vectors = extractor.keypoint_vectors()
        center_vectors = extractor.center_vectors()

        # if self._dataset_config['noise_cfg']['status'] and add_front:
        #     img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + \
        #         front[:, rmin:rmax, cmin:cmax] * \
        #         ~(mask_front[rmin:rmax, cmin:cmax])

        # if desig[:8] == 'data_syn':
        #     img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        cloud = cloud.transpose([2, 0, 1]) # H x W x C -> C x H x W
        img = (np.array(img).astype(np.float32) / 127.5 - 1.0).transpose([2, 0, 1])
        keypoint_vectors = keypoint_vectors.transpose([2, 0, 1])
        center_vectors = center_vectors.transpose([2, 0, 1])
        tup = (torch.from_numpy(cloud.astype(np.float32)),
               torch.from_numpy(img),
               torch.from_numpy(label.astype(np.int)),
               torch.from_numpy(keypoint_vectors),
               torch.from_numpy(center_vectors))

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
                for o in lookup_desig_to_obj[d]:

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

        elif self._dataset_config['batch_list_cfg']['mode'] == 'train_inc_syn':
            desig_ls = self.get_desig(
                self._env_config['p_ycb_seq_train_inc_syn'])

        elif self._dataset_config['batch_list_cfg']['mode'] == 'test':
            desig_ls = self.get_desig(self._env_config['p_ycb_seq_test'])
        else:
            raise AssertionError

        # this is needed to add noise during runtime
        self._syn = self.get_desig(self._env_config['p_ycb_syn'])
        self._real = self.get_desig(self._env_config['p_ycb_seq_train'])
        name = str(self._dataset_config['batch_list_cfg'])
        name = name.replace("""'""", '')
        name = name.replace(" ", '')
        name = name.replace(",", '_')
        name = name.replace("{", '')
        name = name.replace("}", '')
        name = name.replace(":", '')
        name = self._env_config['p_ycb_config'] + '/' + name + '.pkl'
        try:
            with open(name, 'rb') as f:
                batch_ls = pickle.load(f)
        except:
            batch_ls = self.convert_desig_to_batch_list(desig_ls, lookup_dict)

            pickle.dump(batch_ls, open(name, "wb"))

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

