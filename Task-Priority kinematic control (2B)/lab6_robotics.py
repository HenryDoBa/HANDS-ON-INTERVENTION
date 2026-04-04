from lab4_robotics import *

class MobileManipulator:
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
        self.revoluteExt = np.concatenate([np.array([True, False]), self.revolute])        # List of joint types extended with base joints
        self.r = 0.15                                                                       # Distance from robot centre to manipulator base
        self.dof = len(self.revoluteExt)                                                   # Number of DOF of the system
        self.q = np.zeros((len(self.revolute),1))                                          # Vector of joint positions (manipulator)
        self.eta = np.zeros((3,1))                                                         # Vector of base pose (position & orientation)
        self.update(np.zeros((self.dof,1)), 0.0)                                           # Initialise robot state

    '''
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    '''
    def update(self, dQ, dt):
        # Update manipulator
        self.q += dQ[2:, 0].reshape(-1,1) * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i][0]
            else:
                self.d[i] = self.q[i][0]

        # Update mobile base pose

        # First move forward. then rotate.
        #self.eta[0][0] += dQ[1][0] * np.cos(self.eta[2][0]) * dt
        #self.eta[1][0] += dQ[1][0] * np.sin(self.eta[2][0]) * dt
        #self.eta[2][0] += dQ[0][0] * dt

        # First rotate, then move forward. 
        #self.eta[2][0] += dQ[0][0] * dt
        #self.eta[0][0] += dQ[1][0] * np.cos(self.eta[2][0]) * dt
        #self.eta[1][0] += dQ[1][0] * np.sin(self.eta[2][0]) * dt

        # Simultaneous rotation and translation
        if np.abs(dQ[0][0]) < 1e-6:
            self.eta[0][0] += dQ[1][0] * np.cos(self.eta[2][0]) * dt
            self.eta[1][0] += dQ[1][0] * np.sin(self.eta[2][0]) * dt
        else:
            self.eta[0][0] += (dQ[1][0] / dQ[0][0]) * (np.sin(self.eta[2][0] + dQ[0][0] * dt) - np.sin(self.eta[2][0]))
            self.eta[1][0] += (dQ[1][0] / dQ[0][0]) * (-np.cos(self.eta[2][0] + dQ[0][0] * dt) + np.cos(self.eta[2][0]))
        
        self.eta[2][0] += dQ[0][0] * dt


        # Base kinematics
        R_b = np.array([[np.cos(self.eta[2][0]), -np.sin(self.eta[2][0]), 0],
                        [np.sin(self.eta[2][0]), np.cos(self.eta[2][0]), 0],
                        [0, 0, 1]])
        Tb = np.array([[R_b[0, 0], R_b[0, 1], R_b[0, 2], self.eta[0][0]],
                       [R_b[1, 0], R_b[1, 1], R_b[1, 2], self.eta[1][0]],
                       [R_b[2, 0], R_b[2, 1], R_b[2, 2], 0],
                       [0, 0, 0, 1]])           # Transformation of the mobile base



        # Combined system kinematics (DH parameters extended with base DOF)
        dExt = np.concatenate([np.array([0, self.r]), self.d])
        thetaExt = np.concatenate([np.array([np.pi/2, 0]), self.theta])
        aExt = np.concatenate([np.array([0, 0]), self.a])
        alphaExt = np.concatenate([np.array([np.pi/2, -np.pi/2]), self.alpha])

        self.T = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

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
        return self.q[joint-2]


    def getBasePose(self):
        return self.eta

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

    ###
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revoluteExt, link)

    def getLinkTransform(self, link):
        return self.T[link]
