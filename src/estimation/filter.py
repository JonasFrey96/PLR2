import math
import numpy as np 
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from estimation.state import State_SE3, State_R3xQuat
from estimation.motion import Linear_Motion
from scipy.io import loadmat
from glob import glob

def load_ycb_sequence(path, filter):
    """ Loads ycb sequence .mat files into a trajectory. Input is a sequence
    folder destination and a filter object you want to load it into."""
    files = glob(path+'*.mat')
    
    trajectory = []
    bounding_box = []
    
    for f in sorted(files):
        data = loadmat(f)
        f_state_SE3 = State_SE3( np.concatenate(( data['rotation_translation_matrix'], np.array([[0, 0, 0, 1]]) ), axis=0) )
        if filter.state_type=='State_R3xQuat':
            f_state_quat = State_R3xQuat(f_state_SE3)
            trajectory.append(f_state_quat)
        elif filter.state_type=='State_SE3':
            trajectory.append(f_state_SE3)
        elif filter.state_type !='State_SE3':
            raise Exception('Filter state type must be State_SE3 or State_R3xQuat.')
        bounding_box.append(data['center'])
    
    trajectory = tuple(trajectory)
    filter.trajectory = trajectory
    filter.observations = deepcopy(trajectory)
    filter.bounding_box = tuple(bounding_box)
    
    return filter

class Filter( object ):
    def __init__(self, state_type='State_SE3', motion_model='', trajectory=None, observations=None,
                 bounding_box = None, add_noise=False, variance=None):
        self.state_type = state_type
        self.observations = observations
        self.trajectory = trajectory
        self.predictions = None
        self.bounding_box = bounding_box
        self.add_noise_flag = add_noise
        if variance is None:
            self.variance = (0, 0, 0, 0, 0, 0) #workaround for really annoying python thing
    
    def add_noise(self, variance):
        observations = list(self.observations)
        for i in range(1,len(observations)):
            observations[i].add_noise(variance)
        self.observations = tuple(observations)

    def prediction(self, t):
        return

    def update(self, t, measurement):
        return
    
    def pose_error(self, **kwargs):
        """Calculates the error between prediction and ground truth for time step t or whole trajectory."""
        return
    
    def relative_point_error(self, points):
        return

class Linear_Estimator( Filter ):
    """Class to conduct linear estimation. Currently only implemented for 1 object
    with a set of poses from time 0 to end.
    """
    def __init__(self, state_type = 'State_SE3', motion_model = 'Linear_Motion', trajectory = None, 
                 observations = None, variance=0):
        self.state_type = state_type
        self.observations = observations
        self.trajectory = trajectory
        self.predictions = None
        # self.points = np.empty(3, 0) ## might change later to contain points - but points also change over time? 
            
        if motion_model == 'Linear_Motion':
            self.motion_model = Linear_Motion(state_type)
        else:
            raise Exception('Only linear_Motion model supported for Linear_Estimator.')
    
    def predict(self, t=-1):
        """Provides a prediction for the trajectory given a time step"""
        if t==0:
            M = np.identity(4)
        else:
            M = self.motion_model.interpolate_motion(self.observations, t)
        
        return self.motion_model.extrapolate_motion(self.observations, t, M)
            
    def predict_trajectory(self):
        """Interpolates a trajectory based on the observation for input time step.
        Note that the initial prior is assumed to be exact, and the motion for time step 0
        is the identity."""
        motion = [State_SE3()]
        predictions = [self.trajectory[0]]
            
        for t in range(1, len(self.trajectory)):
            motion = self.motion_model.interpolate_motion(self.observations, t-1)
            predictions.append(self.motion_model.extrapolate_motion(self.observations, t-2, motion))
            
        self.predictions = predictions
        
class Kalman_Filter( Filter ):
    """Class to conduct Kalman Filtering. Currently only implemented for 1 object
    with a set of poses from time 0 to end.
    """
    def __init__(self, state_type = 'State_R3xQuat', motion_model = 'Linear_Motion', trajectory = None, 
                 observations = None,
                 variance_motion = (0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1),
                 variance_sensor = (0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1),
                 params=None):
        self.state_type = state_type
        self.observations = observations
        self.trajectory = trajectory
        self.predictions = None
        self.updates = None
        self.covariance_motion = np.diag(variance_motion)
        self.covariance_sensor = np.diag(variance_sensor)
        self.covariances = []
        # self.points = np.empty(3, 0) ## might change later to contain points - but points also change over time? 
            
        if motion_model == 'Linear_Motion':
            self.motion_model = Linear_Motion(state_type, params)
        else:
            raise Exception('Only linear_Motion model supported for Kalman Filter as of now.')
    
    def predict(self, t=-1, covariance_motion=None):
        """Provides a prediction for the trajectory given a time step"""
        M = self.motion_model.interpolate_motion(self.predictions, t)
        pred = self.motion_model.extrapolate_motion(self.predictions, t, M) # model interpolation
        pred.normalize_quat()
        if covariance_motion is None:
            covariance_motion = self.covariance_motion
        elif len(covariance_motion)==7:
            covariance_motion = np.diag(covariance_motion)
        A = self.motion_model.motion_Jacobian(self.predictions[t], M)
        P = np.matmul(A, np.matmul(self.covariances[t], np.transpose(A))) + covariance_motion # prediction covariance
        if t==-1 or t == len(self.observations) - 1:
            self.predictions.append(pred)
            self.covariances.append(P)
        
        return pred, P
    
    def set_prior(self, prior, covariance):
        """Sets a prior for predictions."""
        if self.state_type=='State_R3xQuat':
            self.predictions = [State_R3xQuat(prior)]
        else:
            raise Exception('Only State_R3xQuat state type supported as of now.')
        if len(covariance)==7:
            covariance = np.diag(covariance)
        elif covariance.shape != (7,7):
            raise Exception('Size must be 7 or 7x7.')
        self.covariances = [np.array(covariance)]

    def update(self, observation, covariance_sensor=None):
        """Conducts an update and returns the update value for the last time step.
        Note: ignores observation model calculation H as it is identity matrix."""
        if covariance_sensor is None:
            R = self.covariance_sensor
        elif len(covariance_sensor) == 7:
            R = np.diag(covariance_sensor)
        elif type(covariance_sensor) == np.ndarray and covariance_sensor.shape == (7,7):
            R = covariance_sensor
        else:
            R = self.covariance_sensor
        if isinstance(observation, State_R3xQuat) or isinstance(observation, State_SE3):
            observation = observation.state
        elif not isinstance(observation, np.ndarray):
            observation = np.array(observation).reshape(7,1)
        H = np.identity(7) # Jacobian of measurement model - it's just ones as we have exact observations
        P = self.covariances[-1]
        K = np.linalg.multi_dot([P, np.transpose(H), np.linalg.inv(np.linalg.multi_dot([H, P, np.transpose(H)]) + R)])
        # K = np.matmul(P, np.matmul(np.transpose(H), np.linalg.inv(np.matmul(np.matmul(H, P), np.transpose(H)) + R))) # Kalman gain
        obs_res =  observation - self.predictions[-1].state
        pred = self.predictions[-1].state + np.matmul(K, obs_res)
        # print('pred', pred)
        # print('K', K)

        pred = State_R3xQuat(pred)
        pred.normalize_quat() ## need to normalise quaternion so it is valid
        P = np.multiply((np.identity(7) - np.multiply(K,H)), P)  # adjust covariance
        self.predictions[-1] = pred
        self.covariances[-1] = P
        return pred, P
        
    def predict_trajectory(self, covariance_motion=None, covariance_sensor=None):
        """Interpolates a trajectory based on the observation for input time step.
        Note that the initial prior is assumed to be exact, and the motion for time step 0
        is the identity."""
        self.predictions = [self.observations[0]]
        self.predict(t=-1, covariance_motion=[10, 10, 10, 10, 10, 10, math.pi*20])
        self.prefiltered_predictions = [self.predictions[-1]]

        for t in range(1, len(self.trajectory)-1):
            self.update(self.observations[t], covariance_sensor=covariance_sensor)
            self.predict(t=-1, covariance_motion=covariance_motion)
            self.prefiltered_predictions.append(self.predictions[-1])