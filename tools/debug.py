import argparse
import os
import numpy as np
import math
import torch
import sys
import re
import torch
sys.path.insert(0, os.getcwd())
print(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/src'))
sys.path.append(os.path.join(os.getcwd() + '/lib'))
from tqdm import tqdm
from torch.utils.data import DataLoader
from loaders_v2 import GenericDataset, ConfigLoader
from lib.network import KeypointNet
from lib.loss import KeypointLoss
from visu import helper_functions as vis_helper
from PIL import Image

K = 8

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="Environment yml file to use.")
    parser.add_argument('--exp', type=str)
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--examples', type=int, default=1)
    parser.add_argument('--workers', '-w', type=int, default=1)
    return parser.parse_args()

def convert_to_image(img):
    return ((img + 1.0) * 127.5).cpu().numpy().astype(np.uint8).transpose([1, 2, 0])

def main():
    flags = read_args()

    exp = ConfigLoader().from_file(flags.exp)
    env = ConfigLoader().from_file(flags.env)

    estimator = KeypointNet(**exp.get('net', {}))
    estimator = estimator.cuda()

    dataset = GenericDataset(
        cfg_d=exp['d_test'],
        cfg_env=env)

    dataset = torch.utils.data.Subset(dataset, np.arange(1, 1 + flags.examples))

    parameters = estimator.parameters()
    parameters = {}
    for name, parameter in estimator.named_parameters():
        parameters[name] = parameter.detach().cpu().numpy()

    optimizer = torch.optim.Adam(estimator.parameters(), lr=flags.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
            patience=50, min_lr=1e-7, verbose=True)

    os.makedirs('out/', exist_ok=True)

    loader = DataLoader(dataset, batch_size=flags.batch_size, num_workers=flags.workers, shuffle=True, pin_memory=True)
    loss_fn = KeypointLoss()

    for i in range(1000):
        loader_iter = iter(loader)
        for batch in loader_iter:
            (points, img, label, vertmap, gt_keypoints, gt_centers, cam,
                    objects_in_scene, unique_desig) = batch[0]

            points, img, label, vertmap, gt_keypoints, gt_centers, cam, objects_in_scene = [
                    t.cuda() for t in [points, img, label, vertmap, gt_keypoints, gt_centers, cam, objects_in_scene]]
            predicted_keypoints, object_centers, segmentation = estimator(img, points, vertmap, label)
            loss, (kl, cl, sl) = loss_fn(predicted_keypoints, object_centers, segmentation,
                    gt_keypoints, gt_centers, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss, kl, cl, sl = loss.item(), kl.item(), cl.item(), sl.item()
            print(f"Step {i} loss: {loss:.3f} keypoint: {kl:.3f} center: {cl:.3f} semantic: {sl:.3f}", end='\r')

        scheduler.step(loss)

    with torch.no_grad():
        estimator = estimator.eval()
        (points, img, label, vertmap, gt_keypoints, gt_centers, cam,
                objects_in_scene, unique_desig) = next(iter(loader))[0]

        points, img, label, vertmap, gt_keypoints, gt_centers, cam, objects_in_scene = [
                t.cuda() for t in [points, img, label, vertmap, gt_keypoints, gt_centers, cam, objects_in_scene]]
        predicted_keypoints, object_centers, segmentation = estimator(img, points, vertmap, label)

        for i in range(img.shape[0]):
            H = 240
            W = 320
            image = convert_to_image(img[i])
            camera = cam[i].detach().cpu().numpy()
            current_label = label[i].detach().cpu().numpy()

            object_ids = np.sort(np.unique(current_label))[1:]
            object_index = np.random.randint(0, object_ids.size)
            object_id = object_ids[object_index]
            vote_image = image.copy()
            p_keypoints = predicted_keypoints[i].reshape(8, 3, H, W)
            votes = points[i][None] + p_keypoints
            votes = votes.detach().cpu().numpy().transpose([2, 3, 0, 1])
            vote_image = vis_helper.visualize_votes(vote_image, votes, current_label, object_id, *camera)

            Image.fromarray(vote_image).save(f'out/votes_{i}.jpg')
            seg_image = vis_helper.segmentation_image(current_label)
            Image.fromarray(seg_image).save(f'out/segmentation_{i}.jpg')


    for name, parameter in estimator.named_parameters():
        diff = np.abs(parameters[name] - parameter.detach().cpu().numpy()).sum()
        print(f"Parameter {name} changed {diff}")

if __name__ == "__main__":
    main()

