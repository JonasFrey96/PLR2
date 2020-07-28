
import os
import sys
import copy
from PIL import Image
import pickle as pkl
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from rotations import quat_to_rot
import time
import glob


def get_rot_vec(R):
    x = R[:, 2, 1] - R[:, 1, 2]
    y = R[:, 0, 2] - R[:, 2, 0]
    z = R[:, 1, 0] - R[:, 0, 1]

    r = torch.norm(torch.stack([x, y, z], dim=1))
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    phi = torch.atan2(r, t - 1)
    return phi


def angle_gen(mat, n_mat):
    """
    mat target dim: 3X3
    n_mat dim: Nx3x3

    returns distance betweem the rotation matrixes dim: N
    """
    dif = []
    for i in range(n_mat.shape[0]):
        r, _ = cv2.Rodrigues(mat.dot(n_mat[i, :, :].T))
        dif.append(np.linalg.norm(r))

    return np.array(dif)


def angle_batch_torch_full(mat, n_mat):
    """
    mat target dim: BSx3X3
    n_mat dim: BSxNx3x3

    return BSXN
    """
    bs = mat.shape[0]
    rep = n_mat.shape[1]
    mat = mat.unsqueeze(1).repeat((1, rep, 1, 1))
    mat = mat.view((-1, 3, 3))
    n_mat = n_mat.view((-1, 3, 3))
    out = torch.bmm(mat, torch.transpose(n_mat, 1, 2)).view(-1, 3, 3)

    vectors = get_rot_vec(out).view(bs, -1, 1)
    vectors = torch.abs(vectors)
    idx_argmin = torch.argmin(vectors, dim=1)
    return idx_argmin


class ViewpointManager():

    def __init__(self, store, name_to_idx, device='cuda:0', load_images=False):
        self.store = store
        self.device = device
        self.name_to_idx = name_to_idx
        self.idx_to_name = {}
        self.load_images = load_images

        for key, value in self.name_to_idx.items():
            self.idx_to_name[value] = key

        self._load()
        if self.load_images:

            self._load_images()

    def _load(self):
        self.img_dict = {}
        self.pose_dict = {}
        self.cam_dict = {}
        self.depth_dict = {}
        self.sim_dict = {}

        for obj in self.name_to_idx.keys():
            idx = self.name_to_idx[obj]
            self.pose_dict[idx] = torch.tensor(pkl.load(
                open(f'{self.store}/{obj}/pose.pkl', "rb"))).type(torch.float32).cuda()
            self.cam_dict[idx] = torch.tensor(
                pkl.load(open(f'{self.store}/{obj}/cam.pkl', "rb"))).type(torch.float32).cuda()

    def _load_images(self):

        ls = glob.glob(f'{self.store}/*/*.png')
        self.images = {}
        print('Loading all rendered images. This might take a minute')
        for i, f in enumerate(ls):
            if i % 5000 == 0:
                print(f'Loaded {i}/{len(ls)} images')
            self.images[f] = Image.open(f)
        print('Loaded all Images for ViewpointManger')

    def get_closest_image(self, idx, mat):
        """
        idx: start at 1 and goes to num_obj!
        """
        st = time.time()
        dif = angle_gen(mat, self.pose_dict[idx][:, :3, :3].cpu().numpy())
        idx_argmin = np.argmin(np.abs(dif))

        print('single image idx', idx_argmin, 'value', dif[idx_argmin])
        st = time.time()
        obj = self.idx_to_name[idx]

        st = time.time()
        if self.load_images:
            img = self.images[f'{self.store}/{obj}/{idx_argmin}-color.png']
            depth = self.images[f'{self.store}/{obj}/{idx_argmin}-depth.png']
        else:
            img = Image.open(f'{self.store}/{obj}/{idx_argmin}-color.png')
            depth = Image.open(f'{self.store}/{obj}/{idx_argmin}-depth.png')

        target = self.pose_dict[idx][idx_argmin, :3, :3]
        return self.pose_dict[idx][idx_argmin],
        self.cam_dict[idx][idx_argmin],
        img,
        depth, target, idx_argmin

    def get_closest_image_single(self, idx, mat):
        idx = idx.unsqueeze(0).unsqueeze(0)
        mat = mat.unsqueeze(0)
        return self.get_closest_image_batch(idx, mat)

    def get_closest_image_batch(self, i, rot, conv='wxyz'):
        """
        mat: BSx3x3
        idx: BSx1 0-num_obj-1
        """
        # adapt index notation to 1-num_obj
        idx = copy.copy(i) + 1

        if rot.shape[-1] == 3:
          # rotation matrix input
            pass
        elif rot.shape[-1] == 4:
            rot = quat_to_rot(rot=rot, conv=conv, device=self.device)
        else:
            raise Exception('invalidae shape received for rot', rot.shape)

        sr = self.pose_dict[int(idx[0])].shape  # shape reference size sr
        bs = idx.shape[0]

        # tensor created during runtime to handle flexible batch size
        n_mat = torch.empty((idx.shape[0], sr[0], 3, 3), device=self.device)

        for i in range(0, idx.shape[0]):
            n_mat[i] = self.pose_dict[int(idx[i])][:, :3, :3]

        best_match_idx = angle_batch_torch_full(rot, n_mat)

        img = []
        depth = []
        target = []

        imgls = torch.empty((idx.shape[0], 480, 640, 3), device=self.device)
        depls = torch.empty((idx.shape[0], 480, 640), device=self.device)
        tarls = torch.empty((idx.shape[0], 4, 4), device=self.device)

        st = time.time()
        for j, i in enumerate(idx.tolist()):
            best_match = int(best_match_idx[j])
            obj = self.idx_to_name[i[0]]

            if self.load_images:
                imgls[j, :, :, :] = torch.from_numpy(np.array(
                    self.images[f'{self.store}/{obj}/{best_match}-color.png']).astype(np.float32)).to(self.device).unsqueeze(0)
                depls[j, :, :] = torch.from_numpy(np.array(
                    self.images[f'{self.store}/{obj}/{best_match}-depth.png']).astype(np.float32)).to(self.device).unsqueeze(0)

            else:
                imgls[j, :, :, :] = torch.from_numpy(np.array(Image.open(
                    f'{self.store}/{obj}/{best_match}-color.png')).astype(np.float32)).to(self.device).unsqueeze(0)
                depls[j, :, :] = torch.from_numpy(np.array(Image.open(
                    f'{self.store}/{obj}/{best_match}-depth.png')).astype(np.float32)).to(self.device).unsqueeze(0)

            tarls[j, :, :] = copy.deepcopy(
                self.pose_dict[i[0]][best_match, :4, :4])

        # print(f'loading time is {time.time()-st}')
        return imgls, depls, tarls
