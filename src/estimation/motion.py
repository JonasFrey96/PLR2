import numpy as np
from estimation.state import State_SE3, State_R3xQuat
from scipy.spatial.transform import Rotation as R

class Motion( object ):
    def __init__(self, state_type='State_SE3', trajectory = None):
        self.state_type = state_type
        self.trajectory = trajectory
        
    def interpolate_motion(self, t, time_steps):
        return
    
    def extrapolate_motion(self, t, time_steps):
        return
    
class Linear_Motion( Motion ):
    def __init__(self, state_type='State_SE3', params=None):
        self.state_type = state_type
        if params is None:
            self.a_t = 1.0
            self.a_r = 1.0
        else:
            if 'a_t' in params:
                self.a_t = params['a_t']
            if 'a_r' in params:
                self.a_r = params['a_r']
        
    def interpolate_motion(self, trajectory, t):
        if t==0 or len(trajectory) < 2:
            if self.state_type=='State_SE3':
                return State_SE3()
            elif self.state_type=='State_R3xQuat':
                return State_R3xQuat()
        else:
            t0 = trajectory[t-1]
            t1 = trajectory[t]
            M = t0.relative_transform(t1)
            if isinstance(M, State_SE3):
                M = State_R3xQuat(M)
            id_M = State_R3xQuat() # identity motion
            M.state[0:3,0] = (1-self.a_t)*id_M.state[0:3,0] + self.a_t*M.state[0:3,0] # as the motion transform is relative initial translation is 0,0,0
            theta = np.arccos(np.dot(id_M.state[3:7,0], M.state[3:7,0]))
            quat_slerp = np.true_divide(np.sin((1-self.a_r)*theta)*id_M.state[3:7,0] 
                                        + np.sin(self.a_r*theta)*M.state[3:7,0], np.sin(theta))
            M.state[3:7,0] = quat_slerp
            if self.state_type=='State_SE3':
                M = State_SE3(M)
            return M
        
    def extrapolate_motion(self, trajectory, t, M):
        if isinstance(trajectory[t], State_SE3) or isinstance(trajectory[t], State_R3xQuat):
            return trajectory[t].apply_transform(M)
        else:
            raise Exception('Unsupported state type.')
        
    def motion_Jacobian(self, pose, motion):
        if self.state_type=='State_SE3' or isinstance(pose, State_SE3) or isinstance(motion, State_SE3):
            raise Exception('Jacobians for SE3 not implemented as of now.')
        if self.state_type=='State_R3xQuat' and isinstance(pose, State_R3xQuat) and isinstance(motion, State_R3xQuat):
            return pose.Jacobian_apply_transform(motion)
        else:
            raise Exception('Incorrect input: check both poses are of type State_R3xQuat.')
    
    