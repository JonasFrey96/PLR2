import argparse
import os
import numpy as np
import math
import time
import torch
import sys
import re
from tqdm import tqdm
from loaders_v2 import GenericDataset, ConfigLoader
from lib.network import KeypointNet
from lib.loss import SingleObjectADDLoss
from lib import keypoint_helper
from visu import chw_helper_functions as vis_helper
from torch import multiprocessing
from PIL import Image

K = 8

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default = '', help='Path to checkpoint in run directory to evaluate.')
    parser.add_argument('--env', type=str, help="Environment yml file to use.")
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    return parser.parse_args()

def load_state(estimator, path):
    state = torch.load(path)
    state_dict = {}
    for key, value in state['state_dict'].items():
        if key.index('estimator.') == 0:
            key = key.replace('estimator.', '')
        state_dict[key] = value
    estimator.load_state_dict(state_dict)
    return estimator

def compute_auc(add_values):
    D = np.array(add_values)
    max_distance = 0.1
    D[np.where(D > max_distance)] = np.inf
    D = np.sort(D)
    N = D.shape[0]
    cumulative = np.cumsum(np.ones((1, N))) / N
    return vo_cap(D, cumulative) * 100.0

def vo_cap(D, prec):
    indices = np.where(D != np.inf)
    if len(indices[0]) == 0:
        return 0.0
    D = D[indices]
    prec = prec[indices]
    mrec = np.array([0.0] + mrec.tolist() + [0.1])
    mprec = np.array([0.0] + prec.tolist() + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mprec[i] = max(mprec[i], mprec[i-1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[i] - mrec[i-1]) * mprec[i]) * 10

def save_image(out_dir, item):
    image, object_keypoints, cam, index = item
    for keypoints in object_keypoints:
        points = vis_helper.project_points(keypoints, *cam)
        image = vis_helper.project_keypoints_onto_image(image, points)

    image = image.transpose([1, 2, 0])
    Image.fromarray(image).save(os.path.join(out_dir, f"kp_{index:06d}.png"))

def _image_loop(queue, out_directory):
    while True:
        """
        tuple of items come in from the queue. None is poison pill.
        image: 3 x H x W - numpy array
        object_keypoints: list of 3 x K - numpy arrays
        cam: cx, cy, fx, fy - camera calibration numpy array
        index: int
        """
        item = queue.get()
        if item is None:
            return 0
        save_image(out_directory, item)

def save_seg(out_dir, seg, index):
    seg = seg.round().astype(np.uint8)
    seg = vis_helper.segmentation_image(seg)
    seg_image = seg.transpose([0, 2, 3, 1])
    for i, image in enumerate(seg_image):
        num = index + i
        path = os.path.join(out_dir, f"seg_{num:06d}.png")
        Image.fromarray(image).save(path)

def _seg_loop(queue, out_directory):
    while True:
        item = queue.get()
        if item is None:
            return 0
        seg, index = item
        save_seg(out_directory, seg, index)

class Evaluator:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.voting = keypoint_helper.VotingModule(method='mean_shift', bandwidth=0.05, max_iter=300)
        self.single_object_add = SingleObjectADDLoss()
        self.add = {}
        self.adds = {}
        self.image_queue = multiprocessing.SimpleQueue()
        self.seg_queue = multiprocessing.SimpleQueue()
        self.n_processes = 2
        self._launch_image_process()

    def _launch_image_process(self):
        image_dir = os.path.join(self.out_dir, 'eval_images')
        os.makedirs(image_dir, exist_ok=True)
        self.processes = []
        for _ in range(self.n_processes):
            process = multiprocessing.Process(target=_image_loop, args=(self.image_queue, image_dir))
            process.start()
            self.processes.append(process)
            process = multiprocessing.Process(target=_seg_loop, args=(self.seg_queue, image_dir))
            process.start()
            self.processes.append(process)

    def __call__(self, points, img, p_keypoints, gt_keypoints, gt_label, model_keypoints, object_models,
            objects_in_scene, cam, index):
        gt_keypoints = points[:, None] + gt_keypoints
        p_keypoints = points[:, None] + p_keypoints
        N = p_keypoints.shape[0]

        images = self._to_images(img)

        for i in range(N):
            indices = torch.nonzero(objects_in_scene[i]).flatten()
            image_keypoints = []
            for object_index in indices:
                object_index = object_index.item()
                object_id = object_index + 1
                object_mask = gt_label[i] == object_id
                object_keypoints = p_keypoints[i, :, :, object_mask]
                gt_object_keypoints = gt_keypoints[i, :, :, object_mask]
                keypoints = model_keypoints[object_index]
                object_keypoints = self.voting(object_keypoints)
                gt_object_keypoints = gt_object_keypoints[None, :, :].mean(dim=3)

                gt_T = keypoint_helper.solve_transform(gt_object_keypoints,
                        keypoints)[0]
                T_hat = keypoint_helper.solve_transform(object_keypoints[None],
                        keypoints)[0]
                add = self.single_object_add.asymmetric(gt_T, T_hat, object_models[object_index])
                add_s = self.single_object_add.symmetric(gt_T, T_hat, object_models[object_index])

                if object_id not in self.add:
                    self.add[object_id] = []
                    self.adds[object_id] = []
                self.add[object_id].append(add.item())
                self.adds[object_id].append(add_s.item())

                image_keypoints.append(object_keypoints.cpu().numpy().T)

            self.image_queue.put((images[i], image_keypoints, cam[i].cpu().numpy(), index + i))

    def save_segmentation(self, p_seg, index):
        p_seg = p_seg.argmax(dim=1).cpu().numpy()
        self.seg_queue.put((p_seg, index))

    def _to_images(self, img):
        return ((img + 1.0) * 127.5).cpu().numpy().round().astype(np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for process in self.processes:
            self.image_queue.put(None)
            self.seg_queue.put(None)
        time.sleep(2)
        for process in self.processes:
            process.join()
            process.terminate()

def do_evaluation(out_dir, dataset, model, flags):
    with Evaluator(out_dir) as evaluator:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size,
                shuffle=False, num_workers=2,
                pin_memory=True)

        model_keypoints = dataset.keypoints().cuda()
        object_models = dataset.object_models().cuda()
        i = 0
        with torch.no_grad():
            i = 0
            for batch in tqdm(dataloader):
                i += 1
                batch = [b.cuda() for b in batch[0][:8]]
                (points, img, label, vertmaps, gt_keypoints, gt_centers, cam,
                        objects_in_scene) = batch
                predicted_keypoints, object_centers, segmentation = model(img, points, vertmaps)

                N, _, H, W = predicted_keypoints.shape
                predicted_keypoints = predicted_keypoints.reshape(N, K, 3, H, W)
                gt_keypoints = gt_keypoints.reshape(N, K, 3, H, W)

                evaluator.save_segmentation(segmentation, i)
                add_loss = evaluator(points, img, predicted_keypoints, gt_keypoints, label, model_keypoints,
                        object_models, objects_in_scene, cam, i)

                i += points.shape[0]

        return evaluator.add, evaluator.adds

def main():
    flags = read_args()

    model_dir = os.path.dirname(flags.model)
    exp_file = os.path.join(model_dir, 'exp.yml')
    if flags.env is not None:
        env_file = flags.env
    else:
        env_file = os.path.join(model_dir, 'env.yml')
    exp = ConfigLoader().from_file(exp_file)
    env = ConfigLoader().from_file(env_file)

    estimator = KeypointNet(**exp.get('net', {}))
    estimator = load_state(estimator, flags.model)
    estimator = estimator.cuda().eval()

    dataset_test = GenericDataset(
        cfg_d=exp['d_test'],
        cfg_env=env)

    add_losses, adds_losses = do_evaluation(model_dir, dataset_test, estimator, flags)

    print("object\t| ADD auc\t| ADDS auc")
    add_losses = sorted([(k, v)  for k, v in add_losses.items()], key=lambda x: int(x[0]))
    adds_losses = sorted([(k, v)  for k, v in adds_losses.items()], key=lambda x: int(x[0]))
    for (key, add), (adds_key, add_s) in zip(add_losses, adds_losses):
        assert(key == adds_key)
        add = compute_auc(add)
        add_s = compute_auc(add_s)
        print(f"{key:03d}\t| {add:02f}\t| {add_s:02f}\t")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()

