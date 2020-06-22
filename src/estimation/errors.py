import numpy as np
import sys
sys.path.append('../')
from math import pi
from copy import deepcopy
from estimation.motion import Linear_Motion
from estimation.state import State_R3xQuat, State_SE3, points
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import cv2

def rotation_error(pose1, pose2):
    """ Takes 2 quaternions in xyzw format and returns the angle-axis normalised error."""
    if isinstance(pose1, State_R3xQuat):
        quat1 = pose1.state[3:7,0]
    elif len(pose1)==4:
        quat1 = np.squeeze(pose1)
    else:
        raise Exception('Input must be of type State_R3xQuat or np array with dim 4.')

    if isinstance(pose2, State_R3xQuat):
        quat2 = pose2.state[3:7,0]
    elif len(pose2)==4:
        quat2 = np.squeeze(pose2)
    else:
        raise Exception('Input must be of type State_R3xQuat or np array with dim 4.')

    R1 = R.from_quat(quat1).as_matrix()
    R2 = R.from_quat(quat2).as_matrix()
    r, _ = cv2.Rodrigues(R1.dot(R2.T))
    rotation_error_val = np.linalg.norm(r)
    return rotation_error_val

def translation_error(pose1, pose2):
    if isinstance(pose1, State_R3xQuat):
        t1 = pose1.state[0:3,0]
    elif len(pose1)==3:
        t1 = np.squeeze(t1)
    else:
        raise Exception('Input must be of type State_R3xQuat or np array with dim 3.')

    if isinstance(pose2, State_R3xQuat):
        t2 = pose2.state[0:3,0]
    elif len(pose2)==3:
        t2 = np.squeeze(t2)
    else:
        raise Exception('Input must be of type State_R3xQuat or np array with dim 3.')

    error = np.linalg.norm(t1-t2)
    return error


def ADD(model_points, pose1, pose2):
    """Calculates the Average Distance Error (ADD) for pose1 and pose2
    applied to model_points. Inputs are: points (type np.ndarray or points),
    pose1, pose2 type (State_SE3 or State_R3xQuat).
    """
    if isinstance(model_points, np.ndarray):
        model_points = points(model_points)
    elif not isinstance(model_points, points):
        raise Exception('model_points must be of type points or np.ndarray.')

    model_1 = model_points.apply_transform(pose1)
    model_2 = model_points.apply_transform(pose2)
    error = np.mean(np.linalg.norm(model_1.coords - model_2.coords, axis=0))
    return error

def ADDS(model_points, pose1, pose2):
    """Calculates the Average Distance Error (ADD)-Symmetric for pose1 and pose2
    applied to model_points. Inputs are: points (type np.ndarray or points),
    pose1, pose2 type (State_SE3 or State_R3xQuat).
    """
    if isinstance(model_points, np.ndarray):
        model_points = points(model_points)
    elif not isinstance(model_points, points):
        raise Exception('model_points must be of type points or np.ndarray.')

    model_1 = model_points.apply_transform(pose1)
    model_2 = model_points.apply_transform(pose2)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(model_1.coords.T)
    distances, indices = nbrs.kneighbors(model_2.coords.T)
    error = np.mean(distances)
    return error



