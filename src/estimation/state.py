import numpy as np
from scipy.spatial.transform import Rotation as R

class State():
    def __init__(self, dim=None):
        self.state = np.array([])
        if dim is not None:
            self.dim = dim

    def state_to_np(self, val):
        return self.state

    def __str__(self):
        return np.array2string(self.state)
    
    dim = None

class State_SE3( State ):
    def __init__(self, *args):
        self.dim = (4, 4)
        self.state = np.identity(4, dtype=np.float64)
        if len(args) > 0:
            self.update_state(*args)

    def update_state(self, *args):
        if len(args)==1:
            if isinstance(args[0], State_R3xQuat):
                self.state_from_R3xQuat(args[0])
            else:
                if type(args[0])!=np.ndarray:
                    val = np.array(args[0])
                else:
                    val = args[0]
                if val.shape==(4,4):
                    self.state = val
                else:
                    raise Exception('Dimension of state input is incorrect.')
            
        if len(args)==2:
            if type(args[1])!=np.ndarray:
                val = np.array(args[2])
            if val.size==3:
                    self.state[0:3,3] = val.reshape(3,)
            else:
                raise Exception('Translation must have dimension 3.')

            if type(args[2])!=np.ndarray:
                val = np.array(args[2])

            if val.shape==[3,3]:
                self.state[0:3, 0:3] = val
            else:
                raise Exception('Rotation matrix must have dimensionality 3x3.')

    def state_from_R3xQuat(self, R3xQuat):
        state = np.identity(4, dtype=np.float64)
        if isinstance(R3xQuat, State_R3xQuat):
            state[0:3,3] = R3xQuat.state[0:3,0]
            state[0:3,0:3] = R.from_quat(R3xQuat.state[3:, 0]).as_matrix()
            
        elif isinstance(R3xQuat, np.ndarray):
            state[0:3,3] = R3xQuat[0:3]
            state[0:3,0:3] = R.from_quat(R3xQuat[3:]).as_matrix()
            
        else:
            raise Exception('Input object must be of class type State_TxQuat or dim 7.')
        self.state = state
        return

    def add_noise(self, variance):
        """Adds gaussian noise from the variance, where first 3 entries are xyz noise and next 3 entries are euler angle noise."""
        noise = np.identity(4)
        noise[0:3,0:3] = R.from_euler('zyx', np.random.normal([0, 0, 0], variance[3:6]), degrees=False).as_matrix()
        noise[0:3,3] = np.random.normal([0, 0, 0], variance[0:3]).reshape(3,)
        self.state = np.matmul(self.state, noise)

    def apply_transform(self, T):
        if isinstance(T, State_SE3):
            T = T.state
        elif type(T)==np.ndarray:
            if T.shape!=(4,4):
                raise Exception('Transform must be 4x4 SE3 homogeneous matrix.') 
        else:
            raise Exception('Transform input must be of type State_SE3 or 4x4 numpy array')
                
        T2 = State_SE3(np.matmul( self.state, T) )  
        return T2
            
    def relative_transform(self, T):
        """Takes T as the second state."""
        if isinstance(T, State_SE3):
            T = T.state
        elif type(T)==np.ndarray:
            if T.shape!=(4,4):         
                raise Exception('Transform must be 4x4 SE3 homogeneous matrix.')
        T_R = State_SE3(np.matmul(np.linalg.inv(self.state), T))
        return T_R
    
    def inv_pose(self):
        return State_SE3(np.linalg.inv(self.state))

class State_R3xQuat( State ):
    """Quaternion implementation. The format used is [x y z r] where x,y,z are imaginary axis components 
    and r is the real component.
    TODO: create quat unique pose composition equations (applyTransform and relativeTransform).
    NOTE: You will need to remove the case handlers from motion_model."""
    def __init__(self, *args):
        self.dim = (7,1)
        self.state = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64).reshape(7,1)

        if len(args) > 0:
            if isinstance(args[0], State_SE3):
                self.state_from_SE3(args[0])
            elif isinstance(args[0], State_R3xQuat):
                self.state = args[0].state
            else:
                if type(args[0])!=np.ndarray:
                    val = np.array(args[0])
                else:
                    val = args[0]
                if val.size == 7:
                    self.state = np.array(args[0], dtype=np.float64).reshape(7,1)
                else:
                    raise Exception('Size of quaternion numpy array must be 7.')

    def __str__(self):
       string = 'Rotation Matrix: \n'
       
       string += str(R.from_quat(self.state[3:7,0]).as_matrix() )+ '\n'

       string += 'Quaternion XYZW: '
       string += str(R.from_quat(self.state[3:7,0]).as_quat()) + '\n'

       #string = 'RPY:'
       #string += R.from_quat(self.state).as_matrix()
       string += 'Translation:'
       string += str( self.state[0:3,0])
       return string

    def apply_transform(self, pose2):
        """Applies pose 2 onto this pose and returns another a State_R3xQuat object."""
        return State_R3xQuat(State_SE3(self).apply_transform(State_SE3(pose2)))
    
    def inv_pose(self):
        """Provides inverse Pose of itself as a State_R3xQuat."""
        return State_R3xQuat(State_SE3(self).inv_pose())
    
    def relative_transform(self, pose2):
        """Finds the relative transform between itself and pose2."""
        return self.inv_pose().apply_transform(pose2)
    
    def Jacobian_apply_transform(self, pose2):
        Jac_p1_p2__p1 = np.zeros((7,7), dtype=np.float64)
        h = np.finfo(np.float64).eps

        for i in range(0,7):
            self_h_pos = State_R3xQuat(self.state)
            self_h_pos.state[i,0] += h
            self_h_neg = State_R3xQuat(self.state)
            self_h_neg.state[i,0] -= h
            Jac_col = np.true_divide((self_h_pos.apply_transform(pose2).state - self_h_neg.apply_transform(pose2).state), 2*h)
            Jac_p1_p2__p1[:,i] = np.squeeze(Jac_col)
        return Jac_p1_p2__p1     
    
    def normalize_quat(self):
        quat = self.state[3:7,0]
        self.state[3:7,0] = np.true_divide(quat, np.linalg.norm(quat))
    
    def state_from_SE3(self, SE3):
        state = np.empty((7,1), dtype=np.float64)
        if isinstance(SE3, State_SE3):
            state[0:3] = SE3.state[0:3,3].reshape(3,1)
            state[3:] = R.from_matrix(SE3.state[0:3,0:3]).as_quat().reshape(4,1)
            
        elif isinstance(SE3, np.ndarray):
            state[0:3] = SE3[0:3,3].reshape(3,1)
            state[3:] = R.from_matrix(SE3[0:3,0:3]).as_quat().reshape(4,1)
            
        else:
            raise Exception('Input object must be of class type State_SE3 or equivalent numpy array with dim 4x4.')
        self.state = state
        return

    def state_from_R3xYPR(self, val):
        """Sets state from R3xYPR"""
        state = np.identity(4)
        state[0:3,0:3] = R.from_euler('zyx', np.array(val[3:6]), degrees=False).as_matrix()
        state[0:3,3] = np.array(val[0:3]).reshape(3,)
        self.state_from_SE3(state)

    @staticmethod
    def quat_covariance_from_euler_angle(cov, ypr):
        """Format of covariance and angles must be yaw-pitch-roll."""
        phi = ypr[0] # yaw
        chi = ypr[1] # pitch
        psi = ypr[2] # roll
        
        ccc = np.cos(psi/2)*np.cos(chi/2)*np.cos(phi/2)
        ccs = np.cos(psi/2)*np.cos(chi/2)*np.sin(phi/2)
        csc = np.cos(psi/2)*np.sin(chi/2)*np.cos(phi/2)
        css = np.cos(psi/2)*np.sin(chi/2)*np.sin(phi/2)
        scc = np.sin(psi/2)*np.cos(chi/2)*np.cos(phi/2)
        scs = np.sin(psi/2)*np.cos(chi/2)*np.sin(phi/2)
        ssc = np.sin(psi/2)*np.sin(chi/2)*np.cos(phi/2)
        sss = np.sin(psi/2)*np.sin(chi/2)*np.sin(phi/2)
        J_q_ypr = np.array([[-(csc + scs)/2, -(ssc + ccs)/2, (ccc + sss)/2],
                            [(scc - css)/2, (ccc - sss)/2, (ccs - ssc)/2],
                            [(ccc + sss)/2, -(css + scc)/2, (-csc + scs)/2],
                            [(ssc - ccs)/2, (scs - csc)/2, (css - scc)/2]])
        cov = np.array(cov)
        cov_quat = np.matmul(J_q_ypr, np.matmul(cov, J_q_ypr))
        return cov_quat        

    def add_noise(self, variance):
        """Adds gaussian noise from the variance, where first 3 entries are xyz noise and next 3 entries are euler angle noise."""
        noise = np.identity(4)
        noise[0:3,0:3] = R.from_euler('zyx', np.random.normal([0, 0, 0], variance[3:6]), degrees=False).as_matrix()
        noise[0:3,3] = np.random.normal([0, 0, 0], variance[0:3]).reshape(3,)
        self.state_from_SE3(np.matmul(State_SE3(self).state, noise))


class points():
    def __init__(self, coords=None):
        if type(coords) != np.ndarray:
            coords = np.array(coords)
        
        ## assumes points are organised as column vector
        if coords.shape[1]==3:
            self.coords = coords.T
        elif coords.shape[0]==3:
            self.coords = coords
        else:
            raise Exception('Dimensionality of points must be 3.')
    
    def apply_transform(self, pose):
        """ Returns coordinates of all points in world fixed frame. Returns a new points object."""
        if isinstance(pose, State_R3xQuat):
            pose = State_SE3(pose)

        coords = np.concatenate((self.coords, np.tile([[1]], self.coords.shape[1])), axis=0) # converts to homogenous coordinates
        coords_new = []
        for i in range(0, coords.shape[1]):
            coords_new.append(np.dot(pose.state, coords[:,i]))

        coords_new = np.array(coords_new)
        return points(coords_new[:,:3])

    def __str__(self):
        return np.array2string(self.coords)


if __name__ == "__main__":
    s =State_R3xQuat([0,0,0,0,0,0,1])
    print(type(s)) 
    print(s)
