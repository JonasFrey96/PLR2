import numpy as np
import sys
import os
from PIL import Image
from visu.helper_functions import *
from visu import helper_functions
from scipy.spatial.transform import Rotation as R
from helper import re_quat
import copy
import torch
import numpy as np
import k3d
from matplotlib import cm

KEYPOINT_COLORS = np.array([
    [  0,   0, 127],
    [  0,   0, 255],
    [  0, 128, 255],
    [ 21, 255, 225],
    [124, 255, 121],
    [228, 255,  18],
    [255, 148,   0],
    [255,  29,   0]], dtype=np.uint8)
CENTER_COLOR = np.array([255, 0, 0], dtype=np.uint8)

jet = cm.get_cmap('jet')
SEG_COLORS = (np.stack([jet(v) for v in np.linspace(0, 1, 22)]) * 255).astype(np.uint8)

def project_points(points, cam_cx, cam_cy, cam_fx, cam_fy):
    """
    points: P x 3
    returns: P x 2 in image cordinates
    """
    out = np.zeros((points.shape[0], 2), dtype=np.int32)
    p_x = points[:, 0]
    p_y = points[:, 1]
    p_z = points[:, 2]
    out[:, 0] = (p_x / p_z * cam_fx + cam_cx).astype(np.int32)
    out[:, 1] = (p_y / p_z * cam_fy + cam_cy).astype(np.int32)
    return out

class Visualizer():
    def __init__(self, p_visu, writer=None):
        if p_visu[-1] != '/':
            p_visu = p_visu + '/'
        self.p_visu = p_visu
        self.writer = writer

        if not os.path.exists(self.p_visu):
            os.makedirs(self.p_visu)

    def plot_keypoints(self, tag, epoch, img, points, keypoints, label, store=False,
            cam_cx=0, cam_cy=0, cam_fx=0, cam_fy=0, w=2):
        """
        Project keypoints onto the image and save it. Each keypoint will have it's own color.
        keypoints: H x W x K x 3
        """
        H, W, K, _ = keypoints.shape
        keypoints = points[:, :, None, :] + keypoints.reshape(H, W, K, 3)
        object_ids = np.unique(label)
        for object_index in object_ids[1:]:
            local_image = helper_functions.visualize_votes(img.copy(), keypoints,
                    label,
                    object_index,
                    cam_cx, cam_cy, cam_fx, cam_fy,
                    w=w)
            image_tag = tag + f"_obj{object_index}"
            if store:
                save_image(local_image, tag=f"epoch_{epoch}_{image_tag}",
                        p_store=self.p_visu)
            if self.writer is not None:
                self.writer.add_image(image_tag, local_image, global_step=epoch, dataformats="HWC")

    def plot_centers(self, tag, epoch, img, points, centers, label,
            cam_cx, cam_cy, cam_fx, cam_fy, store=False, w=5):
        """
        Project object centers onto the image and save it.
        centers: H x W x 3
        """
        H, W, _ = centers.shape
        object_ids = np.unique(label)
        local_image = img.copy()
        centers = points + centers
        for object_index in object_ids[1:]:
            mask = label == object_index
            object_centers = centers[mask, :].reshape(-1, 3)
            projected = project_points(object_centers, cam_cx, cam_cy, cam_fx, cam_fy)
            for p in range(projected.shape[0]):
                u, v = projected[p, :]
                try:
                    local_image[v - w:v + w + 1, u - w:u + w + 1, :] = CENTER_COLOR
                except IndexError:
                    pass
        if store:
            save_image(local_image, tag=f"epoch_{epoch}_{tag}",
                    p_store=self.p_visu)
        if self.writer is not None:
            self.writer.add_image(tag, local_image, global_step=epoch, dataformats="HWC")

    def plot_segmentation(self, tag, epoch, label, store):
        if label.dtype == np.float32:
            label = label.round()
        image_out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        for h in range(label.shape[0]):
            for w in range(label.shape[1]):
                image_out[h, w, :] = SEG_COLORS[label[h, w]][:3]

        if store:
            save_image(image_out, tag=f"epoch_{epoch}_{tag}", p_store=self.p_visu)
        if self.writer is not None:
            self.writer.add_image(tag, image_out, global_step=epoch, dataformats="HWC")

    def plot_estimated_pose(self, tag, epoch, img, points, trans=[[0, 0, 0]], rot_mat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], cam_cx=0, cam_cy=0, cam_fx=0, cam_fy=0, store=False, jupyter=False, w=2):
        """
        tag := tensorboard tag
        epoch := tensorboard epoche
        store := ture -> stores the image to standard path
        path := != None creats the path and store to it path/tag.png
        img:= original_image, [widht,height,RGB]
        points:= points of the object model [length,x,y,z]
        trans: [1,3]
        rot: [3,3]
        """

        img_d = copy.deepcopy(img)
        points = np.dot(points, rot_mat.T)
        points = np.add(points, trans[0, :])
        projected = project_points(points, cam_cx, cam_cy, cam_fx, cam_fy)
        for i in range(0, points.shape[0]):
            u, v = projected[i]
            try:
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
                img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = 255
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
            except IndexError:
                #print("out of bounce")
                pass

        if jupyter:
            display(Image.fromarray(img_d))

        if store:
            save_image(img_d, tag=str(epoch) + tag, p_store=self.p_visu)
        if self.writer is not None:
            self.writer.add_image(tag, img_d.astype(
                np.uint8), global_step=epoch, dataformats='HWC')

    def plot_bounding_box(self, tag, epoch, img, rmin=0, rmax=0, cmin=0, cmax=0, str_width=2, store=False, jupyter=False, b=None):
        """
        tag := tensorboard tag
        epoch := tensorboard epoche
        store := ture -> stores the image to standard path
        path := != None creats the path and store to it path/tag.png
        img:= original_image, [widht,height,RGB]

        """

        if isinstance(b, dict):
            rmin = b['rmin']
            rmax = b['rmax']
            cmin = b['cmin']
            cmax = b['cmax']

        # ToDo check Input data
        img_d = np.array(copy.deepcopy(img))

        c = [0, 0, 255]
        rmin_mi = max(0, rmin - str_width)
        rmin_ma = min(img_d.shape[0], rmin + str_width)

        rmax_mi = max(0, rmax - str_width)
        rmax_ma = min(img_d.shape[0], rmax + str_width)

        cmin_mi = max(0, cmin - str_width)
        cmin_ma = min(img_d.shape[1], cmin + str_width)

        cmax_mi = max(0, cmax - str_width)
        cmax_ma = min(img_d.shape[1], cmax + str_width)

        img_d[rmin_mi:rmin_ma, cmin:cmax, :] = c
        img_d[rmax_mi:rmax_ma, cmin:cmax, :] = c
        img_d[rmin:rmax, cmin_mi:cmin_ma, :] = c
        img_d[rmin:rmax, cmax_mi:cmax_ma, :] = c
        print("STORE", store)
        img_d = img_d.astype(np.uint8)
        if store:
            #store_ar = (img_d* 255).round().astype(np.uint8)
            save_image(img_d, tag=str(epoch) + tag, p_store=self.p_visu)
        if jupyter:
            display(Image.fromarray(img_d))
        if self.writer is not None:
            self.writer.add_image(tag, img_d.astype(
                np.uint8), global_step=epoch, dataformats='HWC')


def plot_pcd(x, point_size=0.005, c='g'):
    """
    x: point_nr,3
    """
    if c == 'b':
        k = 245
    elif c == 'g':
        k = 25811000
    elif c == 'r':
        k = 11801000
    elif c == 'black':
        k = 2580
    else:
        k = 2580
    colors = np.ones(x.shape[0]) * k
    plot = k3d.plot(name='points')
    plt_points = k3d.points(x, colors.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


def plot_two_pcd(x, y, point_size=0.005, c1='g', c2='r'):
    if c1 == 'b':
        k = 245
    elif c1 == 'g':
        k = 25811000
    elif c1 == 'r':
        k = 11801000
    elif c1 == 'black':
        k = 2580
    else:
        k = 2580

    if c2 == 'b':
        k2 = 245
    elif c2 == 'g':
        k2 = 25811000
    elif c2 == 'r':
        k2 = 11801000
    elif c2 == 'black':
        k2 = 2580
    else:
        k2 = 2580

    col1 = np.ones(x.shape[0]) * k
    col2 = np.ones(y.shape[0]) * k2
    plot = k3d.plot(name='points')
    plt_points = k3d.points(x, col1.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points = k3d.points(y, col2.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


class SequenceVisualizer():
    def __init__(self, seq_data, images_path, output_path=None):
        self.seq_data = seq_data
        self.images_path = images_path
        self.output_path = output_path

    def plot_points_on_image(self, seq_no, frame_no, jupyter=False, store=False, pose_type='filtered'):
        seq_data = self.seq_data
        images_path = self.images_path
        output_path = self.output_path
        frame = seq_data[seq_no][frame_no]
        unique_desig = frame['dl_dict']['unique_desig'][0]

        if pose_type == 'ground_truth':
            # ground truth
            t = frame['dl_dict']['gt_trans'].reshape(1, 3)
            rot_quat = re_quat(copy.deepcopy(
                frame['dl_dict']['gt_rot_wxyz'][0]), 'wxyz')
            rot = R.from_quat(rot_quat).as_matrix()
        elif pose_type == 'filtered':
            # filter pred
            t = np.array(frame['filter_pred']['t']).reshape(1, 3)
            rot_quat = re_quat(copy.deepcopy(
                frame['filter_pred']['r_wxyz']), 'wxyz')
            rot = R.from_quat(rot_quat).as_matrix()
        elif pose_type == 'final_pred_obs':
            # final pred
            t = np.array(frame['final_pred_obs']['t']).reshape(1, 3)
            rot_quat = re_quat(copy.deepcopy(
                frame['final_pred_obs']['r_wxyz']), 'wxyz')
            rot = R.from_quat(rot_quat).as_matrix()
        else:
            raise Exception('Pose type not implemented.')

        w = 2
        if type(unique_desig) != str:
            im = np.array(Image.open(
                images_path + unique_desig[0] + '-color.png'))  # ycb
        else:
            im = np.array(Image.open(
                images_path + unique_desig + '.png'))  # laval
        img_d = copy.deepcopy(im)

        dl_dict = frame['dl_dict']
        points = copy.deepcopy(
            seq_data[seq_no][0]['dl_dict']['model_points'][0, :, :])
        points = np.dot(points, rot.T)
        points = np.add(points, t[0, :])

        cam_cx = dl_dict['cam_cal'][0][0]
        cam_cy = dl_dict['cam_cal'][0][1]
        cam_fx = dl_dict['cam_cal'][0][2]
        cam_fy = dl_dict['cam_cal'][0][3]
        for i in range(0, points.shape[0]):
            p_x = points[i, 0]
            p_y = points[i, 1]
            p_z = points[i, 2]
            u = int(((p_x / p_z) * cam_fx) + cam_cx)
            v = int(((p_y / p_z) * cam_fy) + cam_cy)
            try:
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
                img_d[v - w:v + w + 1, u - w:u + w + 1, 1] = 255
                img_d[v - w:v + w + 1, u - w:u + w + 1, 0] = 0
            except:
                #print("out of bounds")
                pass

        img_disp = Image.fromarray(img_d)
        if jupyter:
            display(img_disp)

        if store:
            outpath = output_path + \
                '{}_{}_{}.png'.format(pose_type, seq_no, frame_no)
            img_disp.save(outpath, "PNG", compress_level=1)
            print("Saved image to {}".format(outpath))

    def save_sequence(self, seq_no, pose_type='filtered', name=''):
        for fn in range(len(self.seq_data)):
            self.plot_points_on_image(seq_no, fn, False, True, pose_type)
        if name:
            video_name = '{}_{}_{}'.format(name, pose_type, seq_no)
        else:
            video_name = '{}_{}'.format(pose_type, seq_no)
        cmd = "cd {} && ffmpeg -r 10 -i ./filtered_{}_%d.png -vcodec mpeg4 -y {}.mp4".format(
            self.output_path, seq_no, video_name)
        os.system(cmd)

