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
        self.v_ff = np.zeros((0, 1)) # feedforward velocity (initialized as zero)
        self.K = np.eye(0) # feedback gain (initialized as identity)
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
        Method setting the feedforward velocity.
    '''
    def setGain(self, K):
        self.K = K
    '''
        Method returning the feedforward velocity.
    '''
    def getGain(self): 
        return self.K
    '''     
        Method setting the feedforward velocity.
    '''
    def setFeedforward(self, v_ff): 
        self.v_ff = v_ff
    '''     
        Method returning the feedforward velocity.
    '''
    def getFeedforward(self): 
        return self.v_ff


'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, link_index):
        super().__init__(name, desired)
        self.link = link_index # Save link index
        # Initialize v_ff as a 2x1 zero vector
        self.v_ff = np.zeros((2, 1)) 
        # Initialize K as a 2x2 identity matrix
        self.K = np.eye(2)

    def update(self, robot):
        # Get the transformation of the selected link
        T_link = robot.getLinkTransform(self.link)
        current_pos = T_link[0:2, 3].reshape(2, 1)
        
        # Calculate error: err = sigma_d - sigma (desired - current)
        self.err = self.sigma_d - current_pos
        
        # Calculate Jacobian for the selected link
        self.J = robot.getLinkJacobian(self.link)[0:2, :]
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, link_index): 
        super().__init__(name, desired)
        self.link = link_index 
        # Initialize v_ff as a 1x1 zero vector
        self.v_ff = np.zeros((1, 1)) 
        # Initialize K as a 1x1 identity matrix
        self.K = np.eye(1) 

    def update(self, robot):
        # Get the transformation of the selected link
        T_link = robot.getLinkTransform(self.link)
        current_ori = np.arctan2(T_link[1, 0], T_link[0, 0])
        
        self.err = (self.sigma_d - current_ori).reshape(1, 1)
        
        # Get the Jacobian row 5 (angular velocity around Z) for the selected link 
        self.J = robot.getLinkJacobian(self.link)[5:6, :]
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, link_index): 
        super().__init__(name, desired)
        self.link = link_index
        # Initialize v_ff as a 3x1 zero vector
        self.v_ff = np.zeros((3, 1)) 
        # Initialize K as a 3x3 identity matrix
        self.K = np.eye(3) 

    def update(self, robot):
        T_link = robot.getLinkTransform(self.link)
        pos = T_link[0:2, 3]
        ori = np.arctan2(T_link[1, 0], T_link[0, 0])
        
        current_config = np.array([pos[0], pos[1], ori]).reshape(3, 1)
        # Calculate error: err = sigma_d - sigma (desired - current)
        self.err = self.sigma_d - current_config 
        
        # Get the Jacobian for the selected link and extract rows for x, y, and orientation
        full_J = robot.getLinkJacobian(self.link)
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
