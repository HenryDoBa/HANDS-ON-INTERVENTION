# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.array([0, 0, 0])                             # displacement along Z-axis
q = np.array([0.2, 0.5, 0.2]).reshape(3,1)          # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.4])                      # displacement along X-axis
alpha = np.array([0, 0, 0])                          # rotation around X-axis 
revolute = [True, True, True]                        # flags specifying the type of joints


K = np.diag([1, 1]) # Control gain matrix


# Setting desired position of end-effector to the current one
T = kinematics(d, q.flatten(), a, alpha) # flatten() needed if q defined as column vector !
sigma_d = T[-1][0:2,3].reshape(2,1)

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Lists to store history for plotting
q_history = []
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
    global q, a, d, alpha, revolute, sigma_d
    global PPx, PPy, q_history, time_history
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)
    
    # Update control
    sigma = T[-1][0:2, 3].reshape(2, 1)           # Current position of the end-effector
    err = sigma_d - sigma                         # Error in position

    # Calculate Null space projector: P = I - pinv(J) * J
    Jbar = np.linalg.pinv(J)                     # Task Jacobian
    P = np.eye(3) - Jbar @ J                     # Null space projector

    y = np.array([[np.sin(t)], [np.cos(t)], [np.sin(2*t)]])      # Arbitrary joint velocity

    dq = Jbar @ (K @ err) + P @ y                  # Control signal

    # Euler integration to update joint positions
    q = q + dq * dt  # Simulation update 

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    # Save data for final plot
    q_history.append(q.flatten().tolist())
    time_history.append(t)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 60, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()

# After simulation, plot joint positions as required by the lab
q_hist = np.array(q_history)
plt.figure()
for i in range(q_hist.shape[1]):
    plt.plot(time_history, q_hist[:, i], label=f'q{i+1}')
plt.title('Joint Positions')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid(True)
plt.show()