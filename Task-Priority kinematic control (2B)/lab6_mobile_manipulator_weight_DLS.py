from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model
d = np.zeros(3)                            # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])          # rotation around Z-axis
alpha = np.zeros(3)                        # rotation around X-axis
a = np.array([0.4, 0.3, 0.2])              # displacement along X-axis
revolute = np.array([True, True, True])    # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)
damping_factor = 0.1

# Task definition
joint_limit_1 = np.array([-0.5, 0.5])
# Case 1: Equal weights
#W = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
# Case 2: Joints 3, 4, 5 are heavily penalized
#W = np.diag([1.0, 1.0, 10.0, 10.0, 10.0])
# Case 3: Joints 1, 2 (the vehicle base) are heavily penalized
W = np.diag([10.0, 10.0, 1.0, 1.0, 1.0])

tasks = [ 
          Configuration2D("Configuration", np.array([1.0, 0.5, np.pi/2]).reshape(3,1), 5),
        ] 

# Record the data for plotting
error_end_effector = []
error_end_effector_orientation = []
joint_record = {}

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    angle = np.array(np.pi - np.random.rand()*2*np.pi).reshape(1,1)
    desired_position = np.random.rand(2,1)*3-1.5
    tasks[0].setDesired(np.vstack([desired_position, angle])) # Random target
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    
    ### Recursive Task-Priority algorithm
    # Initialize null-space 
    P = np.identity(robot.dof)
    # Initialize output vector (joint velocity)
    dq = np.zeros((robot.dof, 1))
    # Loop over tasks
    for i in range(len(tasks)):

        tasks[i].update(robot)

        if tasks[i].isActivate() != 0:
            # Update task state
            # Padding Jacobian to match the dimension of the robot dof and compute the augmented Jacobian, then compute the DLS solution and accumulate velocity, and finally update the null-space projector
            Jacobian_padded = np.pad(tasks[i].getJacobian(), ((0, 0), (0, robot.dof - tasks[i].getJacobian().shape[1])))
            # Compute augmented Jacobian
            J_bar = Jacobian_padded @ P
            # Task definition
            x_dot = tasks[i].getFeedforwardVelocity() + tasks[i].getGainMatrixK() @ tasks[i].getError()
            # Compute task velocity & Accumulate velocity
            dq = dq + weighted_DLS(J_bar, damping_factor, W) @ (tasks[i].a * x_dot - Jacobian_padded @ dq)
            # Update null-space projector
            P = P - np.linalg.pinv(J_bar) @ J_bar
        else:
            dq = dq
            P = P
    ###

    # Record data for plotting
    error_end_effector.append(np.linalg.norm(tasks[0].getError()[0:2]))
    error_end_effector_orientation.append(abs(tasks[0].getError()[2]))

    for j in range(len(dq)):
        joint_record[j] = joint_record.get(j, []) + [dq[j,0]]

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plotting the end-effector path and distance to the obstacles
tvec = np.arange(0, len(error_end_effector)*dt, dt)
plt.figure()
plt.plot(tvec, error_end_effector, label='e1 (end-effector position)', color='orange')
plt.plot(tvec, error_end_effector_orientation, label='e2 (end-effector orientation)', color='blue')
plt.xlabel('Time[s]')
plt.ylabel('Error[1]')
plt.title('Task-Priority control')
plt.legend()
plt.grid()
plt.show()

# Plotting each joint velocity
plt.figure()
for j in range(len(joint_record)):
    plt.plot(tvec, joint_record[j], label=f'joint {j+1} velocity')
plt.xlabel('Time[s]')
plt.ylabel('Velocity[1]')
plt.title('Task-Priority control')
plt.legend()
plt.grid()
plt.show()