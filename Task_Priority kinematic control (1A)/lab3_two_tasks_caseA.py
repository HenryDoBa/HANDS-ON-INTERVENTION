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
err2_history = []
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
    # TASK 1__End-effector Position
    sigma1 = T[-1][0:2, 3].reshape(2,1)            # Current position of the end-effector
    err1 = sigma1_d - sigma1                       # Error in Cartesian position
    J1 = J[0:2, :]                                 # Task 1 Jacobian
    #Use Damped Least Squares (DLS) for Task 1 inverse
    J1_star = DLS(J1, damping)  
    dq1 = J1_star @ (K1 @ err1)                     # Joint velocity for Task 1
    P1 = np.eye(3) - np.linalg.pinv(J1) @ J1        # Null space projector of Task 1
    
    # TASK 2__Joint 1 Constraint
    sigma2 = np.array([[q[0,0]]])                   # Current position of joint 1
    err2 = sigma2_d - sigma2                        # Error in joint position
    J2 = np.array([[1, 0, 0]])                      # Task 2 Jacobian
    # Project Task 2 Jacobian into the Null Space of task 1
    J2_aug = J2 @ P1 
    J2_aug_star = DLS(J2_aug, damping)              # Augmented Jacobian
    
    # Combining tasks
    dq12 = dq1 + J2_aug_star @ (K2 @ err2 - J2 @ dq1)

    s = np.max(np.abs(dq12) / dq_max)
    if s > 1:
        dq_final = dq12 / s
    else:
        dq_final = dq12

    q = q + dq_final * dt # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    # Record error norms for logging
    err1_history.append(np.linalg.norm(err1))
    err2_history.append(np.linalg.norm(err2))
    time_history.append(t)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_history, err1_history, label='Error Task 1 (EE)')
plt.plot(time_history, err2_history, label='Error Task 2 (Joint 1)')
plt.title('Evolution of Task Errors (CaseA)')
plt.xlabel('Time [s]')
plt.ylabel('Error Norm')
plt.legend()
plt.grid(True)
plt.show()

# Save error data to .npy files for later comparison
np.save('error1_caseA.npy', np.array(err1_history))
np.save('error2_caseA.npy', np.array(err2_history))