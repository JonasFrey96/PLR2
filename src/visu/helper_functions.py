import datetime
import imageio
import numpy as np
from pathlib import Path
from lib import keypoint_helper
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
SEG_COLORS = (np.stack([jet(v)[:3] for v in np.linspace(0, 1, 22)]) * 255).astype(np.uint8)

def save_image(image_array,tag='img',p_store='~/track_this/results/imgs/',time_stamp = False):
    """
    image_array = np.array [width,height,RGB]
    """
    if p_store[-1] != '/':
        p_store = p_store + '/'
    Path(p_store).mkdir(parents=True, exist_ok=True)

    if time_stamp:
        tag = str(datetime.datetime.now().replace(microsecond=0).isoformat())+ tag
    imageio.imwrite( p_store + tag + '.jpg', image_array)

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

def segmentation_image(image):
    """ Colors the segmentation mask for visualization. """
    out_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    for object_id in range(image.max()):
        out_image[image == object_id, :] = SEG_COLORS[object_id]
    return out_image

def project_keypoints_onto_image(image, keypoints, cam, w=3):
    # keypoints: K x 3
    cam_cx, cam_cy, cam_fx, cam_fy = cam
    projected = project_points(keypoints, cam_cx, cam_cy, cam_fx, cam_fy)
    for i in range(projected.shape[0]):
        u, v = projected[i, :]
        color = KEYPOINT_COLORS[i]
        try:
            image[v - w:v + w + 1, u - w:u + w + 1, :] = color
        except IndexError:
            pass

    return image

def compute_keypoints(points, p_keypoints, label):
    """
    points: 3 x H x W
    p_keypoints: K x 3 x H x W
    """
    object_ids = np.sort(np.unique(label)[1:])
    p_keypoints = points[None] + p_keypoints
    keypoints = []
    for object_id in object_ids:
        object_mask = label == object_id
        object_keypoints = p_keypoints[:, :, object_mask]
        object_keypoints = object_keypoints.mean(axis=2)
        keypoints.append(object_keypoints)
    return np.stack(keypoints)

def visualize_votes(image, votes, label, object_id, cam_cx, cam_cy, cam_fx, cam_fy, w=1):
    mask = label == object_id
    K = votes.shape[2]
    object_keypoints = votes[mask, :].reshape(-1, 3)
    projected = project_points(object_keypoints, cam_cx, cam_cy, cam_fx, cam_fy)
    projected = projected.reshape(-1, K, 2)
    for p in range(projected.shape[0]):
        for i in range(K):
            u, v = projected[p, i, :]
            color = KEYPOINT_COLORS[i]
            try:
                image[v - w:v + w + 1, u - w:u + w + 1, :] = color
            except IndexError:
                pass
    return image

def points_to_image(image, projected, color, cam, w=4):
    for i in range(projected.shape[0]):
        u, v = projected[i, :]
        try:
            image[v - w:v + w + 1, u - w:u + w + 1, :] = color
        except IndexError:
            pass
    return image
