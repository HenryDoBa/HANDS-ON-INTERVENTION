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
W = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])

conf1 = np.array([1.5, 0.0, np.pi/2]).reshape(3,1)
conf2 = np.array([1.0, 0.5, 2*np.pi/3]).reshape(3,1)
conf3 = np.array([0.5, 1.0, 5*np.pi/6]).reshape(3,1)
conf4 = np.array([0.0, 1.5, np.pi]).reshape(3,1)
conf5 = np.array([-0.5, 1.0, 7*np.pi/6]).reshape(3,1)
conf6 = np.array([-1.0, 0.5, 4*np.pi/3]).reshape(3,1)
conf7 = np.array([-1.5, 0.0, 3*np.pi/2]).reshape(3,1)

desired_configuration = [conf1, conf2, conf3, conf4, conf5, conf6, conf7]

tasks = [ 
          Configuration2D("Configuration", np.array([1.0, 0.5, np.pi/2]).reshape(3,1), 5),
        ] 

# Counter for switching the target
counter = -3

# Record the data for plotting
error_end_effector = []
error_end_effector_orientation = []
robot_pose = {}

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
    global counter
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    if counter < len(desired_configuration):
        tasks[0].setDesired(desired_configuration[counter]) # Set target
        counter += 1
    return line, path, point

# Simulation loop
def simulate(t):
    global counter
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

    robot_pose['x'] = robot_pose.get('x', []) + [robot.getBasePose()[0,0]]
    robot_pose['y'] = robot_pose.get('y', []) + [robot.getBasePose()[1,0]]

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

# Plotting path of the mobile base and end-effector
plt.figure()
plt.plot(robot_pose['x'], robot_pose['y'], label='Mobile base path', color='red')
plt.plot(PPx, PPy, label='End-effector path', color='purple')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Path of the mobile base and end-effector')
plt.legend()
plt.grid()
plt.show()

# Saving the data for plotting
method_name = "Method_C" 
np.save(f'{method_name}_base_x.npy', robot_pose['x'])
np.save(f'{method_name}_base_y.npy', robot_pose['y'])
np.save(f'{method_name}_ee_x.npy', PPx)
np.save(f'{method_name}_ee_y.npy', PPy)