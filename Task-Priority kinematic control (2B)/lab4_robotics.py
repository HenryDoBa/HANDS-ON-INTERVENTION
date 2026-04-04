from lab2_robotics import * # Includes numpy import
from scipy.spatial.transform import Rotation as R

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    J = np.zeros((6, link))
    O = T[-1][0:3, 3]
    # 2. For each joint of the robot
    for i in range(0, link):
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
            z = np.array([z]).T
            J_i = np.vstack((z, np.zeros((3, 1))))
        #   c. Modify corresponding column of J.
        J[:, i] = J_i[:, 0]
    return J


'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i][0]
            else:
                self.d[i] = self.q[i][0]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]
    
    '''
        Method that returns the transformation of a selected link.
    '''
    def getLinkTransform(self, link):
        return self.T[link]
    
    '''
        Method that returns the Jacobian of a selected link.
    '''
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.a = 0 # Activation value
        self.ff_vel = None
        self.gain_matrix_K = None
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    ''' 
        Method setting the feedforward velocity.

        Arguments:
        value(Numpy array): value of the feedforward velocity vector
    '''
    def setFeedforwardVelocity(self, value):
        self.ff_vel = value
    
    '''
        Method returning the feedforward velocity vector.
    '''
    def getFeedforwardVelocity(self):
        return self.ff_vel
    
    ''' 
        Method setting the gain matrix K.

        Arguments:
        value(Numpy array): value of the gain matrix K
    '''
    def setGainMatrixK(self, value):
        self.gain_matrix_K = value
    
    '''
        Method returning the gain matrix K.
    '''
    def getGainMatrixK(self):
        return self.gain_matrix_K

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    '''
        Method checking the active state.
    '''    
    def isActivate(self) -> bool:
        return self.a

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, link_index):
        super().__init__(name, desired)
        self.err = np.zeros(desired.shape) # Initialize with proper dimensions
        self.link_index = link_index # Index of the link for which the task is defined
        self.ff_vel = np.zeros(desired.shape) # Initialize feedforward velocity
        self.gain_matrix_K = np.eye(desired.shape[0]) # Initialize gain matrix K
        self.a = 1 # Initialize activate value of task
        
    def update(self, robot):
        self.J = robot.getLinkJacobian(self.link_index)[0:2,:]   # Update task Jacobian
        self.err = self.getDesired() - robot.getLinkTransform(self.link_index)[0:2, 3].reshape(2,1)# Update task error
        
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, link_index):
        super().__init__(name, desired)
        self.err = 0.0 # Initialize with proper dimensions
        self.link_index = link_index # Index of the link for which the task is defined
        self.ff_vel = np.zeros(1) # Initialize feedforward velocity
        self.gain_matrix_K = np.eye(1) # Initialize gain matrix K
        self.a = 1 # Initialize activate value of task

    def update(self, robot):
        self.J = robot.getLinkJacobian(self.link_index)[5, :].reshape(1, self.link_index) # Update task Jacobian
        rotation_matrix_desired = R.from_euler('z', self.getDesired()).as_matrix() # Convert desired orientation from Euler angles to rotation matrix
        w_d, epsilon_d = self.quaternion_from_rotation_matrix(rotation_matrix_desired)
        w, epsilon = self.quaternion_from_rotation_matrix(robot.getLinkTransform(self.link_index)[0:3, 0:3])
        self.err = (w * epsilon_d - w_d * epsilon - np.cross(epsilon, epsilon_d))[-1]# Update task error
        self.err = np.array(self.err).reshape(1, 1) # Reshape error to be a column vector

    # Method to convert rotation matrix to quaternion (scalar-first convention)
    def quaternion_from_rotation_matrix(self, orientation_matrix):
        orientation_matrix = R.from_matrix(orientation_matrix)
        q = orientation_matrix.as_quat().flatten()
        w = q[3]
        epsilon = q[0:3]

        return w, epsilon
    
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, link_index):
        super().__init__(name, desired)
        self.err = np.zeros(desired.shape) # Initialize with proper dimensions
        self.link_index = link_index # Index of the link for which the task is defined
        self.ff_vel = np.zeros(desired.shape) # Initialize feedforward velocity
        self.gain_matrix_K = np.eye(desired.shape[0]) # Initialize gain matrix K
        self.a = 1 # Initialize activate value of task

    def update(self, robot):
        self.J = np.vstack((robot.getLinkJacobian(self.link_index)[0:2,:], robot.getLinkJacobian(self.link_index)[5, :])).reshape(3, self.link_index)# Update task Jacobian
        rotation_matrix_desired = R.from_euler('z', self.getDesired()[-1]).as_matrix() # Convert desired orientation from Euler angles to rotation matrix
        w_d, epsilon_d = self.quaternion_from_rotation_matrix(rotation_matrix_desired) # Desired orientation in quaternion form
        w, epsilon = self.quaternion_from_rotation_matrix(robot.getLinkTransform(self.link_index)[0:3, 0:3]) # Current orientation in quaternion form
        self.err = np.vstack((self.getDesired()[0:2].reshape(2,1) - robot.getLinkTransform(self.link_index)[0:2, 3].reshape(2,1), 
                              (w * epsilon_d - w_d * epsilon - np.cross(epsilon, epsilon_d))[-1]))# Update task error
        self.err = self.err.reshape(3, 1) # Reshape error to be a column vector

    def quaternion_from_rotation_matrix(self, orientation_matrix):
        orientation_matrix = R.from_matrix(orientation_matrix)
        q = orientation_matrix.as_quat().flatten()
        w = q[3]
        epsilon = q[0:3]

        return w, epsilon

''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, index, desired):
        super().__init__(name, desired)
        self.err = 0.0 # Initialize with proper dimensions
        self.index = index # Index of the joint for which the task is defined
        self.ff_vel = np.zeros(1) # Initialize feedforward velocity
        self.gain_matrix_K = np.eye(1) # Initialize gain matrix K
        self.a = 1 # Initialize activate value of task
      
    def update(self, robot):
        self.J = np.zeros((1, robot.getDOF())) # Update task Jacobian
        self.J[0, self.index] = 1 # Set the column corresponding to the joint index to 1
        self.err = self.getDesired() - robot.getJointPos(self.index) # Update task error
        self.err = np.array(self.err).reshape(1, 1) # Reshape error to be a column vector

''' 
    Subclass of Task, representing the obstacle task.
'''
class Obstacle2D(Task):
    def __init__(self, name, obstacle_pos, obs_boundary):
        super().__init__(name, obstacle_pos)
        self.err = 0.0 # Initialize with proper dimensions
        self.ff_vel = np.zeros(len(obstacle_pos)).reshape(2, 1) # Initialize feedforward velocity
        self.gain_matrix_K = np.eye(len(obstacle_pos)) # Initialize gain matrix K
        self.radius_alpha = obs_boundary[0]
        self.radius_gamma = obs_boundary[1]
        
    def update(self, robot):
        self.J = robot.getLinkJacobian(3)[0:2, :] # Update task Jacobian
        dist_ee_obs = robot.getLinkTransform(3)[0:2, 3].reshape(2,1) - self.getDesired().reshape(2,1)
        self.err = dist_ee_obs/(abs(dist_ee_obs))

        # Task switching logic
        if self.isActivate() == 0 and np.linalg.norm(dist_ee_obs) <= self.radius_alpha:
            self.a = 1
        elif self.isActivate() == 1 and np.linalg.norm(dist_ee_obs) >= self.radius_gamma:
            self.a = 0

''' 
    Subclass of Task, representing the obstacle task.
'''
class JointLimit(Task):
    def __init__(self, name, safe_set, index):
        super().__init__(name, safe_set)
        self.err = 0.0 # Initialize with proper dimensions
        self.index = index
        self.ff_vel = np.zeros(1).reshape(1, 1) # Initialize feedforward velocity
        self.gain_matrix_K = np.eye(1) # Initialize gain matrix K
        self.alpha = 0.02
        self.gamma = 0.05
        
    def update(self, robot):
        self.J = np.zeros((1, robot.getDOF())) # Update task Jacobian
        self.J[0, self.index] = 1 # Set the column corresponding to the joint index to 1
        self.err = np.array([[1]])
        joint_pos = robot.getJointPos(self.index)

        # Task switching logic
        if self.isActivate() == 0 and joint_pos >= (self.getDesired()[1] - self.alpha):
            self.a = -1
        elif self.isActivate() == 0 and joint_pos <= (self.getDesired()[0] + self.alpha):
            self.a = 1
        elif self.isActivate() == -1 and joint_pos <= (self.getDesired()[1] - self.gamma):
            self.a = 0
        elif self.isActivate() == 1 and joint_pos >= (self.getDesired()[0] + self.gamma):
            self.a = 0