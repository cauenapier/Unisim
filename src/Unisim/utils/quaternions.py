import numpy as np

def euler2quat(roll, pitch, yaw):
    '''Given the euler_angles vector, the quaternion vector is returned.

    Parameters
    ----------
    euler_angles : array_like
        1x3 array with the euler angles: [theta, phi, psi]    (rad)

    Returns
    -------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_1, q_2, q_3]

    References
    ----------
    .. [1] "Modeling and Simulation of Aerospace Vehicle Dynamics" (Aiaa\
        Education Series) Peter H. Ziepfel
    '''
    phi = roll
    psi = yaw
    theta = pitch


    q_0 = np.cos(psi/2)*np.cos(theta/2)*np.cos(phi/2) + np.sin(psi/2)*np.sin(theta/2)*np.sin(phi/2)

    q_1 = np.cos(psi/2)*np.cos(theta/2)*np.sin(phi/2) - np.sin(psi/2)*np.sin(theta/2)*np.cos(phi/2)

    q_2 = np.cos(psi/2)*np.sin(theta/2)*np.cos(phi/2) + np.sin(psi/2)*np.cos(theta/2)*np.sin(phi/2)

    q_3 = np.sin(psi/2)*np.cos(theta/2)*np.cos(phi/2) - np.cos(psi/2)*np.sin(theta/2)*np.sin(phi/2)

    quaternion = np.array([q_0, q_1, q_2, q_3])

    return quaternion

def quat2euler(quaternion):
    check_unitnorm(quaternion)
    quaternion_normalized = quaternion/np.linalg.norm(quaternion)
    q_0, q_1, q_2, q_3 = quaternion_normalized


    psi = np.arctan2(2*(q_1*q_2 + q_0*q_3), q_0**2 + q_1**2 - q_2**2 - q_3**2)

    theta = np.arcsin(-2*(q_1 * q_3 - q_0 * q_2))

    phi = np.arctan2(2*(q_2 * q_3 + q_0 * q_1), q_0**2 - q_1**2 - q_2**2 + q_3**2)

    euler_angles = np.array([psi, theta, phi])

    return euler_angles

def quat2DCM(quat):
    """
    """
    DCM = np.zeros((3,3))
    DCM[0,0] = quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2
    DCM[0,1] = 2*(quat[1]*quat[2] + quat[0]*quat[3])
    DCM[0,2] = 2*(quat[1]*quat[3] - quat[0]*quat[2])
    DCM[1,0] = 2*(quat[1]*quat[2] - quat[0]*quat[3])
    DCM[1,1] = quat[0]**2 - quat[1]**2 + quat[2]**2 - quat[3]**2
    DCM[1,2] = 2*(quat[2]*quat[3] + quat[0]*quat[1])
    DCM[2,0] = 2*(quat[1]*quat[3] + quat[0]*quat[2])
    DCM[2,1] = 2*(quat[2]*quat[3] - quat[0]*quat[1])
    DCM[2,2] = quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2

    return DCM

def check_unitnorm(quaternion):
    '''Given a quaternion, it checks the modulus (it must be unit). If it is
    not unit, it raises an error.

    Parameters
    ----------
    quaternion : array_like
        1x4 vector with the four elements of the quaternion:
        [q_0, q_1, q_2, q_3]
    Raises
    ------
    ValueError:
        Selected quaternion norm is not unit
    '''
    q_0, q_1, q_2, q_3 = quaternion
    error = (q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2) - 1

    check_value = np.isclose((q_0 ** 2 + q_1 ** 2 + q_2 ** 2 + q_3 ** 2), [1])

    if not check_value:
        pass
        #print("Quaternion Error:", error)

def quaternion_rotation(quat, vector):
    vector_rot = np.zeros(3)
    #quat_normalized = quat/np.linalg.norm(quat)

    vector_rot[0] = 2*(0.5 - quat[2]**2 - quat[3]**2)*vector[0]
    vector_rot[0] = vector_rot[0] + 2*(quat[1]*quat[2]+quat[0]*quat[3])*vector[1]
    vector_rot[0] = vector_rot[0] + 2*(-quat[0]*quat[2]+quat[1]*quat[3])*vector[2]

    vector_rot[1] = 2*(quat[1]*quat[2]-quat[0]*quat[3])*vector[0]
    vector_rot[1] = vector_rot[1] + 2*(0.5 - quat[1]**2 - quat[3]**2)*vector[1]
    vector_rot[1] = vector_rot[1] + 2*(quat[0]*quat[1]+quat[2]*quat[3])*vector[2]

    vector_rot[2] = 2*(quat[1]*quat[3]+quat[0]*quat[2])*vector[0]
    vector_rot[2] = vector_rot[2] + 2*(-quat[0]*quat[1]+quat[2]*quat[3])*vector[1]
    vector_rot[2] = vector_rot[2] + 2*(0.5 - quat[1]**2 - quat[2]**2)*vector[2]

    return vector_rot
