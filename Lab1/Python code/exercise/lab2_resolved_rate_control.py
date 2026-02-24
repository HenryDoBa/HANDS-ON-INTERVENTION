# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 
revolute = [True, True]
sigma_d = np.array([0.0, 1.0])
K = np.diag([1, 1])

# Simulation params
dt = 1.0/60.0

error_history = []
method_name = "pinv" # Change to "transpose" or "pinv" or "dls" for subsequent runs

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
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
    global d, q, a, alpha, revolute, sigma_d
    global PPx, PPy

    # Update robot
    T = kinematics(d, q, a, alpha)
    J = jacobian(T, revolute) # Implement!

    # Update control
    sigma = T[-1][0:2, 3]          # Position of the end-effector
    err = sigma_d - sigma          # Control error
    error_history.append(np.linalg.norm(err))

    # Option 1: Transpose
    #dq = J.T @ (K @ err)
    
    # Option 2: Pseudoinverse
    dq = np.linalg.pinv(J) @ (K @ err)
    
    # Option 3: Damped Least Squares (DLS)
    #invJ = DLS(J, 0.1)             # damping = 0.1
    #dq = invJ @ (K @ err)          # Control solution

    q += dt * dq
    
    # Update drawing
    P = robotPoints2D(T)
    line.set_data(P[0,:], P[1,:])
    PPx.append(P[0,-1])
    PPy.append(P[1,-1])
    path.set_data(PPx, PPy)
    #point.set_data(sigma_d[0], sigma_d[1])
    point.set_data([sigma_d[0]], [sigma_d[1]])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()

plt.figure()
plt.plot(np.arange(0, len(error_history)*dt, dt), error_history)
plt.title('Evolution of control error norm')
plt.xlabel('Time [s]')
plt.ylabel('Error [m]')
plt.grid(True)
plt.show()

# Saving
print(f"Saving data for {method_name}...")
np.save(f'error_{method_name}.npy', np.array(error_history))