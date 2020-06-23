import numpy as np
import sys
sys.path.append('../')
from math import pi
from copy import deepcopy
from estimation.filter import Linear_Estimator, Kalman_Filter, load_ycb_sequence
from estimation.motion import Linear_Motion
from estimation.state import State_R3xQuat, State_SE3, points
from estimation.errors import ADD, ADDS
from loaders_v2 import ConfigLoader
from scipy.spatial.transform import Rotation as R


def calculate_print_error(pred, gt):
    print('translation error:', np.sum(
        np.abs(pred.state[0:3] - gt.state[0:3])))
    print('rotation error squared:', np.sum(
        np.abs(pred.state[3:] - gt.state[3:7])))


if __name__ == '__main__':
    print('Check SE(3) apply transform.')
    state1_np = np.identity(4)
    state1 = State_SE3(state1_np)

    state1_motion_np = np.identity(4)
    state1_motion_np[0:3, 0:3] = R.from_euler(
        'zyx', np.array([20, 10, 13]), degrees=True).as_matrix()
    state1_motion_np[:, 3] = np.array([1, 0.5, 2, 1]).reshape(4,)
    state1_motion = State_SE3(state1_motion_np)

    print('state1', state1)
    print('state1_motion', state1_motion)

    state2 = state1.apply_transform(state1_motion)
    print('state2', state2)

    print('Recover SE(3) relative transform - relative transform between state 1 and 2 w.r.t state 1 .')
    state1_2 = state1.relative_transform(state2)
    print('state1_2', state1_2.state)

    print('Apply same motion to a third state and check relative transform is the same.')
    state3 = state2.apply_transform(state1_motion)
    state2_3 = state2.relative_transform(state3)

    print('state3 =\n', state3)
    print('state2_3 == state1_2, with error', np.sum(
        abs(state2_3.state - state1_2.state)))

    print('Check State_R3xQuat relative transform.')
    r3xq1 = State_R3xQuat(state1)
    r3xq1_motion = State_R3xQuat()
    r3xq1_motion.state_from_R3xYPR([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])

    print('r3xq1 =\n', r3xq1)
    print('r3xq1_motion =\n', r3xq1_motion)
    print('State_SE3 from r3xq1_motion == state1_motion, with error', np.sum(
        np.abs(State_SE3(r3xq1_motion).state - state1_motion.state)))
    r3xq2 = r3xq1.apply_transform(r3xq1_motion)

    print('r3xq2 =\n', r3xq2)
    r3xq1_2 = r3xq1.relative_transform(r3xq2)
    print('r3xq1_2 =\n', r3xq1_2)

    print('Apply the motion as an State_R3xQuat to a third state and make sure relative motion is the same.')
    r3xq3 = r3xq2.apply_transform(r3xq1_motion)
    r3xq2_3 = r3xq2.relative_transform(r3xq3)
    print('r3xq1_2 == r3xq2_3, with error', np.sum(
        np.abs(r3xq2_3.state - r3xq1_2.state)))

    print('Check Jacobian_apply_transform, for an identity pose motion applied to an identity pose.')
    Jac_no_motion = r3xq1.Jacobian_apply_transform(r3xq1)
    print('r3xq1.Jacobian_apply_transform(r3xq1) =\n', Jac_no_motion)

    print('Check Jacobian_apply_transform, for a r3xq1_motion applied to the identity pose r3xq1.')
    Jac_motion_1_2 = r3xq1.Jacobian_apply_transform(r3xq1_motion)
    print('r3xq1.Jacobian_apply_transform(r3xq1_motion) =\n',
          np.array2string(Jac_motion_1_2, max_line_width=np.inf))

    print('Check Jacobian_apply_transform for r3xq1_motion applied to pose r3xq2.')
    Jac_motion_2_3 = r3xq2.Jacobian_apply_transform(r3xq1_motion)
    print('r3xq2.Jacobian_apply_transform(r3xq1_motion) =\n',
          np.array2string(Jac_motion_2_3, max_line_width=np.inf))

    print('Check Linear_Motion motion model behaviour. First with default settings of a_t = 1.0 and a_r = 1.0')

    trajectory = [r3xq1, r3xq2, r3xq3, r3xq3.apply_transform(r3xq1_motion)]
    LM = Linear_Motion(state_type='State_R3xQuat')

    print('Check interpolate_motion.')
    lm0_1 = LM.interpolate_motion(trajectory, 1)
    print('lm0_1 == r3xq1_2, with error', np.sum(
        np.abs(r3xq1_2.state - lm0_1.state)))
    lm1_2 = LM.interpolate_motion(trajectory, 2)
    print('lm1_2 == r3xq2_3, with error', np.sum(
        np.abs(r3xq2_3.state - lm1_2.state)))

    print('Check extrapolate_motion.')
    lm2_exp = LM.extrapolate_motion(trajectory, 1, lm1_2)
    print('lm2_exp == trajectory[2], with error', np.sum(
        np.abs(lm2_exp.state - trajectory[2].state)))

    print('Check Linear_Motion motion model interpolate behaviour, with a_t = 0.5 and a_r = 0.')
    params_LM2 = {'a_t': 0.5, 'a_r': 0}
    LM2 = Linear_Motion(state_type='State_R3xQuat', params=params_LM2)
    lm1_2_damped = LM2.interpolate_motion(trajectory, 2)
    print('lm1_2_damped translation interpolation comparison error for undamped',
          np.sum(np.abs(lm1_2_damped.state[0:3, 0] - np.true_divide(lm1_2.state[0:3, 0], 2))))
    print('lm1_2_damped rotation', lm1_2_damped.state[3:7, 0])

    print('Testing point pose transforms.')
    points1 = points([[1, 2, 3], [4, 5, 6]])
    points1_t = points1.apply_transform(r3xq1)
    print('Points1 transformed with identity transform is the same with error',
          np.sum(np.abs(points1.coords - points1_t.coords)))

    print('Testing ADD error.')
    print('ADD error for same two poses done on points is',
          ADD(points1, r3xq1_motion, r3xq1_2))

    # symmetric 4 points along z axis
    points1_s = points([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    pose1 = State_R3xQuat()
    # rotation along z axis of 90 degrees
    pose1.state_from_R3xYPR([0, 0, 0, pi / 2, 0, 0])
    pose2 = State_R3xQuat()
    # rotation along z axis of -90 degrees
    pose2.state_from_R3xYPR([0, 0, 0, -pi / 2, 0, 0])

    print('ADD-S error for symmetric points with two poses along axis of symmetry is ',
          ADDS(points1_s, pose1, pose2))

    print('Check Kalman_Filter with Linear_Motion behaviour. Motion variance set high, with observations low.')
    variance_motion_1 = [10, 10, 10, 10, 10, 10, 20 * pi]
    variance_sensor_1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    print('variance_motion_1', variance_motion_1)
    print('variance_sensor_1', variance_sensor_1)
    params_KF = {'a_t': 1.0, 'a_r': 1.0}
    KF1 = Kalman_Filter(state_type='State_R3xQuat',
                        motion_model='Linear_Motion',
                        trajectory=None,
                        observations=None,
                        variance_motion=variance_motion_1,
                        variance_sensor=variance_sensor_1,
                        params=params_KF)

    KF1.set_prior(trajectory[0], variance_motion_1)

    KF1.predict()
    KF1.update(trajectory[1])
    print('KF1 after Update at t=1:')
    calculate_print_error(KF1.predictions[1], trajectory[1])

    KF1.predict()
    print('KF1 after Predict for t=2:')
    calculate_print_error(KF1.predictions[2], trajectory[2])
    KF1.update(trajectory[2])
    print('KF1 after Update at t=2:')
    calculate_print_error(KF1.predictions[2], trajectory[2])

    KF1.predict()
    print('KF1 after Predict for t=3:')
    calculate_print_error(KF1.predictions[3], trajectory[3])

    print('Check Kalman_Filter with Linear_Motion behaviour. Motion are observations at equal variance.')
    a_t = 0.5
    a_r = 0.2
    variance_motion_2 = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1]
    variance_sensor_2 = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    print('variance_motion_2', variance_motion_2)
    print('variance_sensor_2', variance_sensor_2)
    KF2 = Kalman_Filter(state_type='State_R3xQuat',
                        motion_model='Linear_Motion',
                        trajectory=None,
                        observations=None,
                        variance_motion=variance_motion_2,
                        variance_sensor=variance_sensor_2,
                        params=params_KF)

    KF2.set_prior(trajectory[0], variance_motion_1)
    print('diag(KF2.covariances[0]) as prior variances_motion_1 (high),', np.diag(
        KF2.covariances[0]))

    KF2.predict(t=-1, covariance_motion=variance_motion_1)
    KF2.update(trajectory[1], covariance_sensor=[0, 0, 0, 0, 0, 0, 0])
    print('KF2 after Update at t=1, note: covariance at this step set to 0 to allow correct motion model initialisation.')
    calculate_print_error(KF2.predictions[1], trajectory[1])

    KF2.predict()
    print('KF2 after Predict for t=2:')
    calculate_print_error(KF2.predictions[2], trajectory[2])
    KF2.update(trajectory[2])
    print('KF2 after Update at t=2:')
    calculate_print_error(KF2.predictions[2], trajectory[2])

    KF2.predict()
    print('KF2 after Predict for t=3:')
    calculate_print_error(KF2.predictions[3], trajectory[3])
    KF2.update(trajectory[3])
    print('KF2 after Update at t=3:')
    calculate_print_error(KF2.predictions[3], trajectory[3])

    for i in range(4, 50):
        nextPose = trajectory[i - 1].apply_transform(r3xq1_motion)
        trajectory.append(nextPose)
        KF2.predict()
        KF2.update(trajectory[i])

    print('KF after constant motion applied until t={}:'.format(len(trajectory) - 1))
    calculate_print_error(KF2.predictions[-1], trajectory[-1])
