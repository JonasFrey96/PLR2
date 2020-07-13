import copy
from PIL import Image
import pickle as pkl
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2


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

    def __init__(self, store, name_to_idx, device='cuda:0'):
        self.store = store
        self.device = device
        self.name_to_idx = name_to_idx
        self.idx_to_name = {}

        for key, value in self.name_to_idx.items():
            self.idx_to_name[value] = key

        self._load()

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
        img = Image.open(f'{self.store}/{obj}/{idx_argmin}-color.png')
        depth = Image.open(f'{self.store}/{obj}/{idx_argmin}-depth.png')
        target = self.pose_dict[idx][idx_argmin, :3, :3]
        return self.pose_dict[idx][idx_argmin],\
            self.cam_dict[idx][idx_argmin],\
            img,\
            depth, target, idx_argmin

    def get_closest_image_single(self, idx, mat):
        idx = idx.unsqueeze(0).unsqueeze(0)
        mat = mat.unsqueeze(0)
        return self.get_closest_image_batch(idx, mat)

    def get_closest_image_batch(self, idx, mat):
        """
        mat: BSx3x3
        idx: BSx1
        """
        sr = self.pose_dict[int(idx[0])].shape  # shape reference size sr
        bs = idx.shape[0]

        # tensor created during runtime to handle flexible batch size
        n_mat = torch.empty((idx.shape[0], sr[0], 3, 3), device=self.device)

        for i in range(0, idx.shape[0]):
            n_mat[i] = self.pose_dict[int(idx[i])][:, :3, :3]

        calc = time.time()
        best_match_idx = angle_batch_torch_full(mat, n_mat)
        print("Calc time", time.time() - calc)

        img = []
        depth = []
        target = []

        for j, i in enumerate(idx.tolist()):
            best_match = int(best_match_idx[j])
            obj = self.idx_to_name[i[0]]

            img.append(Image.open(
                f'{self.store}/{obj}/{best_match}-color.png'))
            depth.append(Image.open(
                f'{self.store}/{obj}/{best_match}-depth.png'))
            target.append(self.pose_dict[i[0]][best_match, :3, :3])

        return img, depth, target


if __name__ == "__main__":
    # test rotation vector
    from scipy.stats import special_ortho_group

    mat = np.array(special_ortho_group.rvs(dim=3, size=10))
    Rin = torch.from_numpy(mat).type(torch.float32).cuda()
    q = get_rot_vec(Rin)

    # load dataset
    # import os
    import os
    import sys
    os.chdir('/home/jonfrey/PLR')
    sys.path.append('src')
    sys.path.append('src/dense_fusion')

    import scipy.io as scio
    from loaders_v2 import Backend, ConfigLoader, GenericDataset
    import time
    exp_cfg = ConfigLoader().from_file(
        '/home/jonfrey/PLR/src/loaders_v2/test/dataset_cfgs.yml')
    env_cfg = ConfigLoader().from_file(
        '/home/jonfrey/PLR/src/loaders_v2/test/env_ws.yml')

    generic = GenericDataset(
        cfg_d=exp_cfg['d_ycb'],
        cfg_env=env_cfg)

    # load data from dataloader
    model = '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/models'
    base = '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/data/0003'
    desig = '000550'
    store = '/media/scratch1/jonfrey/datasets/YCB_Video_Dataset/viewpoints_renderings'
    img = Image.open('{0}/{1}-color.png'.format(base, desig))
    obj = '025_mug'

    vm = ViewpointManager(store, generic._backend._name_to_idx)

    # apply the same to verify it with an image

    obj_idx = generic._backend._name_to_idx[obj]

    meta = scio.loadmat('{0}/{1}-meta.mat'.format(base, desig))
    obj_tmp = meta['cls_indexes'].flatten().astype(np.int32)
    obj_idx_in_list = int(np.argwhere(obj_tmp == obj_idx))
    target_r = np.array(meta['poses'][:, :, obj_idx_in_list][:, 0:3])
    target_t = np.array(
        [meta['poses'][:, :, obj_idx_in_list][:, 3:4].flatten()])[0, :]

    start = time.time()
    # test numpy implementation
    p, c, img, depth, target, idx_argmin = vm.get_closest_image(
        idx=obj_idx, mat=target_r)
    print("Time get single image numpy cv2 backbone: ", time.time() - start)

    t_target_r = torch.tensor(target_r, dtype=torch.float32).cuda()
    t_obj_idx = torch.tensor(obj_idx, dtype=torch.int64).cuda()

    start = time.time()
    # test pytorch implementation
    img, depth, target = vm.get_closest_image_single(
        idx=t_obj_idx, mat=t_target_r)
    print("Time get single image pytorch: ", time.time() - start)

    bs = 10
    start = time.time()
    img, depth, target = vm.get_closest_image_batch(
        idx=t_obj_idx.view((-1, 1)).repeat((bs, 1)), mat=t_target_r.view(-1, 3, 3).repeat((bs, 1, 1)))
    print(f'Time get {bs} images pytorch: ', time.time() - start)

    # This looks good.
