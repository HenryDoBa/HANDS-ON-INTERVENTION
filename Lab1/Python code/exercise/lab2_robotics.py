import numpy as np # Import Numpy

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    # 2. Multiply matrices in the correct order (result in T).

    # Matrix rotated around z and translated along z
    Tz = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                   [np.sin(theta),  np.cos(theta), 0, 0],
                   [0,               0,            1, d],
                   [0,               0,            0, 1]])
    
    # Matric rotated around x and translated along x
    Tx = np.array([[1, 0,             0,              a],
                   [0, np.cos(alpha), -np.sin(alpha), 0],
                   [0, np.sin(alpha),  np.cos(alpha), 0],
                   [0, 0,             0,              1]])
    
    T = Tz @ Tx 
    return T

def kinematics(d, theta, a, alpha):
    '''
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
    '''
    T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    for i in range(len(d)):
        Ti_local = DH(d[i], theta[i], a[i], alpha[i])
        Ti_global = T[-1] @ Ti_local
        T.append(Ti_global)
    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    n = len(T) - 1 # Number of joints
    J = np.zeros((6, n))
    o_ee = T[-1][0:3, 3]

    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.
    for i in range (n):
        z_prev = T[i][0:3, 2] 
        o_prev = T[i][0:3, 3]

        if revolute[i]: 
            J[0:3, i] = np.cross(z_prev, (o_ee - o_prev)) 
            J[3:6, i] = z_prev 
        else: 
            J[0:3, i] = z_prev
            J[3:6, i] = np.zeros(3)

    return J[[0, 1], :]

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    I = np.eye(A.shape[0])
    A_dls = A.T @ np.linalg.inv(A @ A.T + (damping**2) * I)
    return A_dls # Implement the formula to compute the DLS of matrix A.

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P
