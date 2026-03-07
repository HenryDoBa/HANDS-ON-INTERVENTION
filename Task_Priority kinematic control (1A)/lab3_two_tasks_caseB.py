# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.array([0, 0, 0])                             # displacement along Z-axis
q = np.array([0.2, 0.5, 0.2]).reshape(3,1)          # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.4])                      # displacement along X-axis
alpha = np.array([0, 0, 0])                         # rotation around X-axis 
revolute = [True, True, True]                       # flags specifying the type of joints

# Control Parameters
K1 = np.diag([2.0, 2.0])                       # Gain matrix for Task 1 (End-effector position)
K2 = np.diag([1.5])                            # Gain matrix for Task 2 (Joint 1 constraint)
damping = 0.1                                  # Damping factor for DLS to handle singularities
dq_max = 1.0                                   # Maximum allowed joint velocity

# Desired values of task variables
sigma1_d = np.array([0.0, 1.0]).reshape(2,1) # Position of the end-effector
sigma2_d = np.array([[0.0]]) # Position of joint 1

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Save history
err1_history = []
joint_1_record = []
time_history = []

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy
    
    # Random target at start of each 10s cycle
    if t <= dt:
        sigma1_d = np.random.uniform(-1.2, 1.2, (2, 1))
        #PPx, PPy = [], []                        # Reset path trace for new target

    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    # Update control
    # TASK 2
    sigma2 = np.array([[q[0,0]]])               # Current position of joint 1
    err2 = sigma2_d - sigma2                    # Error in joint position
    J2 = np.array([[1.0, 0.0, 0.0]])            # Task 2 Jacobian 
    #Use Damped Least Squares (DLS)
    J2_star = DLS(J2, damping) 
    dq2 = J2_star @ (K2 @ err2)                 # Joint velocity for Task 2
    P2 = np.eye(3) - np.linalg.pinv(J2) @ J2    # Null space projector of Task 2 

     # TASK 1
    sigma1 = T[-1][0:2, 3].reshape(2, 1)            # Current position of the end-effector
    err1 = sigma1_d - sigma1                        # Error in Cartesian position
    J1 = J[0:2, :]                                  # Task 1 Jacobian
    # Project Task 1 Jacobian into the Null Space of Task 2
    J1_aug = J1 @ P2 
    J1_aug_star = DLS(J1_aug, damping)      # Augmented Jacobian
    
    # Combining tasks
    dq21 = dq2 + J1_aug_star @ (K1 @ err1 - J1 @ dq2)

    s = np.max(np.abs(dq21) / dq_max)
    if s > 1:
        dq_final = dq21 / s
    else:
        dq_final = dq21

    q = q + dq_final * dt # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    # Record error norms for final plotting
    err1_history.append(np.linalg.norm(err1))
    joint_1_record.append(abs(q[0, 0]))
    time_history.append(t)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot error of end-effector and joint 1 position
t_rec = np.arange(len(err1_history)) * dt # Time vector for recorded joint angles
plt.figure(figsize=(8, 4))
plt.plot(t_rec, err1_history, label='e1 (end-effector position)')
plt.plot(t_rec, joint_1_record, label='e2 (joint 1 position)')
plt.title('Task-Priority (two tasks) - CaseB')
plt.xlabel('Time [s]')
plt.ylabel('Error Norm')
plt.legend()
plt.grid(True)
plt.show()