from lab2_robotics import * # Includes numpy import

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
    # Code almost identical to the one from lab2_robotics...
    n = len(revolute)
    J = np.zeros((6, n))
    o_target = T[link][0:3, 3]
    for i in range(link): # Iterate only until the selected link
        z_prev = T[i][0:3, 2]
        o_prev = T[i][0:3, 3]

        if revolute[i]:
            J[0:3, i] = np.cross(z_prev, (o_target - o_prev))
            J[3:6, i] = z_prev
        else:
            J[0:3, i] = z_prev
            J[3:6, i] = np.zeros(3)
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
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
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
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        # Initialize 2D error vector (x, y)
        self.err = np.zeros((2, 1)) 
        # Initialize 2x3 Jacobian for a 3-DOF robot 
        self.J = np.zeros((2, 3))   

    def update(self, robot):
        # Get the current End Effector (EE) position from the transformation matrix T
        T_ee = robot.getEETransform()
        current_pos = T_ee[0:2, 3].reshape(2, 1)
        
        # Calculate error: err = sigma_d - sigma (desired - current)
        self.err = self.sigma_d - current_pos
        
        # Update Jacobian: Extract the first 2 rows from the robot's 6xN Jacobian matrix
        self.J = robot.getEEJacobian()[0:2, :]
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        # Initialize 1D error for orientation (theta)
        self.err = np.zeros((1, 1))
        # Initialize 1x3 Jacobian for a 3-DOF robot
        self.J = np.zeros((1, 3))

    def update(self, robot):
        # Get the End Effector (EE) transformation matrix
        T_ee = robot.getEETransform()
        
        # Calculate current orientation from the rotation matrix (using atan2 on R21 and R11)
        current_ori = np.arctan2(T_ee[1, 0], T_ee[0, 0])
        
        # Compute the orientation error
        self.err = (self.sigma_d - current_ori).reshape(1, 1)
        
        # The Jacobian for orientation corresponds to the angular velocity about the Z-axis (index 5)
        # as some lab setups might only return a 2xN Jacobian (x, y).
        self.J = robot.getEEJacobian()[5:6, :]
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        # Initialize 3D error vector (x, y, theta)
        self.err = np.zeros((3, 1))
        # Initialize 3x3 Jacobian for a 3-DOF robot
        self.J = np.zeros((3, 3))

    def update(self, robot):
        # Get the End Effector (EE) transformation matrix
        T_ee = robot.getEETransform()
        
        # Extract 2D position (x, y)
        pos = T_ee[0:2, 3]
        
        # Calculate current orientation (theta) from the rotation matrix
        ori = np.arctan2(T_ee[1, 0], T_ee[0, 0])
        
        # Combine position and orientation into a single configuration vector
        current_config = np.array([pos[0], pos[1], ori]).reshape(3, 1)
        
        # Calculate total error: err = desired_config - current_config
        self.err = self.sigma_d - current_config
        
        # Construct the task Jacobian by selecting relevant rows
        # Row 0, 1: Linear velocities in X and Y
        # Row 5: Angular velocity around Z (rotation)
        full_J = robot.getEEJacobian()
        self.J = full_J[[0, 1, 5], :]

''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired, joint_index):
        super().__init__(name, desired)
        self.joint_index = joint_index
        # Initialize 1D error for a single joint
        self.err = np.zeros((1, 1))
        # Initialize Jacobian; will be resized during update to match robot DOF
        self.J = np.zeros((1, 3))

    def update(self, robot):
        # Get the current joint position from the robot state
        current_q = robot.getJointPos(self.joint_index)
        
        # Calculate joint error: err = desired_q - current_q
        self.err = (self.sigma_d - current_q).reshape(1, 1)
        
        # The Jacobian for joint i is a unit vector (selection vector) 
        # that pulls only the velocity of joint i.
        self.J = np.zeros((1, robot.getDOF()))
        self.J[0, self.joint_index] = 1.0
