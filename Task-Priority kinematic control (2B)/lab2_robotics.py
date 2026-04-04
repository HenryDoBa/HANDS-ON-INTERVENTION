import numpy as np  # Import Numpy
from math import cos, sin


def DH(d, theta, a, alpha):
    """
    Function builds elementary Denavit-Hartenberg transformation matrices
    and returns the transformation matrix resulting from their multiplication.

    Arguments:
    d (double): displacement along Z-axis
    theta (double): rotation around Z-axis
    a (double): displacement along X-axis
    alpha (double): rotation around X-axis

    Returns:
    (Numpy array): composition of elementary DH transformations
    """
    # 1. Build matrices representing elementary transformations (based on input parameters).
    T_d = np.array([[1, 0, 0, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, 1, d], 
                    [0, 0, 0, 1]]) 
    T_theta = np.array(
        [
            [cos(theta), -sin(theta), 0, 0],
            [sin(theta), cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    T_a = np.array([[1, 0, 0, a], 
                    [0, 1, 0, 0], 
                    [0, 0, 1, 0], 
                    [0, 0, 0, 1]])
    T_alpha = np.array(
        [
            [1, 0, 0, 0],
            [0, cos(alpha), -sin(alpha), 0],
            [0, sin(alpha), cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )
    # 2. Multiply matrices in the correct order (result in T).
    T = T_d @ T_theta @ T_a @ T_alpha
    return T


def kinematics(d, theta, a, alpha, Tb=np.eye(4)):
    """
    Functions builds a list of transformation matrices, for a kinematic chain,
    descried by a given set of Denavit-Hartenberg parameters.
    All transformations are computed from the base frame.

    Arguments:
    d (Numpy array): list of displacements along Z-axis
    theta (Numpy array): list of rotations around Z-axis
    a (Numpy array): list of displacements along X-axis
    alpha (Numpy array): list of rotations around X-axis

    Returns:
    (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    """
    T = [Tb]  # Base transformation
    # For each set of DH parameters:
    for i in range(len(d)):
        # 1. Compute the DH transformation matrix.
        n_1_T_n = DH(d[i], theta[i], a[i], alpha[i])
        # 2. Compute the resulting accumulated transformation from the base frame.
        T_0_n = T[-1] @ n_1_T_n
        # 3. Append the computed transformation to T.
        if i == 1:
            T_0_n = T_0_n @ np.array([[0, 1, 0, 0],
                                      [-1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        T.append(T_0_n)
    return T


# Inverse kinematics
def jacobian(T, revolute):
    """
    Function builds a Jacobian for the end-effector of a robot,
    described by a list of kinematic transformations and a list of joint types.

    Arguments:
    T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

    Returns:
    (Numpy array): end-effector Jacobian
    """
    # 1. Initialize J and O.
    n_dof = len(revolute)
    J = np.zeros((6, n_dof))
    O = T[-1][0:3, 3]
    # 2. For each joint of the robot
    for i in range(n_dof):
        #   a. Extract z and o.
        i_1_T_i = T[i]
        z = i_1_T_i[0:3, 2]
        o = i_1_T_i[0:3, 3]
        #   b. Check joint type.
        if revolute[i] == True:
            cross_vector = np.array([np.cross(z, (O - o))]).T
            z = np.array([z]).T
            J_i = np.vstack((cross_vector, z))
        else:
            J_i = np.vstack((z, np.zeros((3, 1))))
        #   c. Modify corresponding column of J.
        J[:, i] = J_i[:, 0]
    return J


# Damped Least-Squares
def DLS(A, damping):
    """
    Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

    Arguments:
    A (Numpy array): matrix to be inverted
    damping (double): damping factor

    Returns:
    (Numpy array): inversion of the input matrix
    """
    DLS_matrix = A.T @ np.linalg.inv(A @ A.T + damping**2 * np.identity(len(A)))
    return DLS_matrix  # Implement the formula to compute the DLS of matrix A.

# Damped Least-Squares
def weighted_DLS(A, damping, W):
    """
    Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

    Arguments:
    A (Numpy array): matrix to be inverted
    damping (double): damping factor
    weight (Numpy array): weight matrix

    Returns:
    (Numpy array): inversion of the input matrix
    """
    DLS_weighted_matrix = np.linalg.inv(W) @ A.T @ np.linalg.inv(A @ np.linalg.inv(W) @ A.T + damping**2 * np.identity(len(A)))
    return DLS_weighted_matrix  # Implement the formula to compute the DLS of matrix A.


# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    """
    Function extracts the characteristic points of a kinematic chain on a 2D plane,
    based on the list of transformations that describe it.

    Arguments:
    T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)

    Returns:
    (Numpy array): an array of 2D points
    """
    P = np.zeros((2, len(T)))
    for i in range(len(T)):
        P[:, i] = T[i][0:2, 3]
    return P