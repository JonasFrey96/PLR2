import yagmail
from sklearn.neighbors import NearestNeighbors
import yaml
import numpy as np
import collections

import torch
import copy


def flatten_list(d, parent_key='', sep='_'):
    items = []
    for num, element in enumerate(d):
        new_key = parent_key + sep + str(num) if parent_key else str(num)

        if isinstance(element, collections.MutableMapping):
            items.extend(flatten_dict(element, new_key, sep=sep).items())
        else:
            if isinstance(element, list):
                if isinstance(element[0], dict):
                    items.extend(flatten_list(element, new_key, sep=sep))
                    continue
            items.append((new_key, element))
    return items


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, list):
                if isinstance(v[0], dict):
                    items.extend(flatten_list(v, new_key, sep=sep))
                    continue
            items.append((new_key, v))
    return dict(items)


def norm_quat(q):
    # ToDo raise type and dim error
    return q / torch.sqrt(torch.sum(q * q))


def pad(s, sym='-', p='l', length=80):
    if len(s) > length:
        return s
    else:
        if p == 'c':
            front = int((length - len(s)) / 2)
            s = sym * front + s
            back = int(length - len(s))
            s = s + sym * back
        if p == 'l':
            back = int(length - len(s))
            s = s + sym * back
        return s


def re_quat(q, input_format):

    if input_format == 'xyzw':
        if isinstance(q, torch.Tensor):
            v0 = q[0].clone()
        else:
            v0 = copy.deepcopy(q[0])

        q[0] = q[3]
        q[3] = q[2]
        q[2] = q[1]
        q[1] = v0
        return q
    elif input_format == 'wxyz':
        if isinstance(q, torch.Tensor):
            v0 = q[0].clone()
        else:
            v0 = copy.deepcopy(q[0])

        q[0] = q[1]
        q[1] = q[2]
        q[2] = q[3]
        q[3] = v0
        return q


def send_email(text):
    yag = yagmail.SMTP('trackthisplr', "TrackThis")
    contents = [
        "Run is finished!",
        text
    ]
    yag.send('jonfrey@student.ethz.ch',
             'PLR - TrackThis - Lagopus', contents)
    yag.send('yavyas@student.ethz.ch',
             'PLR - TrackThis - Lagopus', contents)


def compose_quat(p, q, device):
    """
    input is wxyz
    """
    q = norm_quat(re_quat(q.squeeze(), 'wxyz')).unsqueeze(0)
    p = norm_quat(re_quat(p.squeeze(), 'wxyz')).unsqueeze(0)
    product = torch.zeros(
        (max(p.shape[0], q.shape[0]), 4), dtype=torch.float32, device=device)
    product[:, 3] = p[:, 3] * q[:, 3] - torch.sum(p[:, :3] * q[:, :3], (1))
    product[:, :3] = (p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] +
                      torch.cross(p[:, :3], q[:, :3]))

    return re_quat(product.squeeze(0), 'xyzw')


def rotation_angle(q, device):
    # in radians
    q = norm_quat(q)
    unit_r = torch.t(torch.tensor(
        [[0, 0, 0, 1]], dtype=torch.float32, device=device))
    return torch.asin(torch.mm(q, unit_r)) * 2


def nearest_neighbor(src, dst):
    assert src.shape[1] == dst.shape[1]

    neigh = NearestNeighbors(n_neighbors=1, n_jobs=8)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def replace_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def generate_unique_idx(num, max_idx):
    a = random.sample(range(0, max_idx), k=min(num, max_idx))
    while len(a) < num:
        a = a + random.sample(
            range(0, max_idx), k=min(max_idx, num - len(a)))
    return a


def get_bbox_480_640(label):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280,
                   320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640

    # print(type(label))
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
