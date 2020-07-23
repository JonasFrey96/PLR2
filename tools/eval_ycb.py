import argparse
import os
import numpy as np
import math
import torch
import sys
import re
from tqdm import tqdm
from loaders_v2 import GenericDataset, ConfigLoader
from lib.network import KeypointNet
from lib.loss import MultiObjectADDLoss

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
    mrec = np.array([0.0] + D.tolist() + [0.1])
    mprec = np.array([0.0] + prec.tolist() + [prec[-1]])
    for i in range(1, prec.shape[0]):
        mprec[i] = max(mprec[i], mprec[i-1])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[i] - mrec[i-1]) * mprec[i]) * 10

def compute_add_scores(dataset, model, flags):
    add_loss = MultiObjectADDLoss()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size,
            shuffle=True, num_workers=0,
            pin_memory=True)

    add_losses = {}
    adds_losses = {}
    model_keypoints = dataset.keypoints().cuda()
    object_models = dataset.object_models().cuda()
    with torch.no_grad():
        i = 0
        for batch in tqdm(dataloader):
            i += 1
            batch = [b.cuda() for b in batch[0][:7]]
            (points, img, label, gt_keypoints, gt_centers, cam,
                    objects_in_scene) = batch
            predicted_keypoints, object_centers, segmentation = model(img, points)

            N, _, H, W = predicted_keypoints.shape
            predicted_keypoints = predicted_keypoints.reshape(N, K, 3, H, W)
            gt_keypoints = gt_keypoints.reshape(N, K, 3, H, W)

            add_loss(points, predicted_keypoints, gt_keypoints, label, model_keypoints,
                    object_models, objects_in_scene, add_losses, adds_losses)
    return add_losses, adds_losses

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

    add_losses, adds_losses = compute_add_scores(dataset_test, estimator, flags)

    print("object\t| ADD auc\t| ADDS auc")
    add_losses = sorted([(k, v)  for k, v in add_losses.items()], key=lambda x: int(x[0]))
    adds_losses = sorted([(k, v)  for k, v in adds_losses.items()], key=lambda x: int(x[0]))
    for (key, add), (adds_key, add_s) in zip(add_losses, adds_losses):
        assert(key == adds_key)
        add = compute_auc(add)
        add_s = compute_auc(add_s)
        print(f"{key:03d}\t| {add:02f}\t| {add_s:02f}\t")


if __name__ == "__main__":
    main()

