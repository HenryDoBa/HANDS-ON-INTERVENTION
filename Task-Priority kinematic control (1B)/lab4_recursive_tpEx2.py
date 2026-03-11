from lab4_roboticsEx2 import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d = np.zeros(3)                      # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.6])    # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.5])       # displacement along X-axis
alpha = np.zeros(3)                  # rotation around X-axis
revolute = [True, True, True]        # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
tasks = [
   Position2D("End-effector position", np.array([1.2, 0.8]).reshape(2,1), 3),
   Orientation2D("Link 2 orientation", np.array([0.0]).reshape(1,1), 2)
]

#Set up the Gain Matrix K for Task 1
tasks[0].setGain(np.diag([1.0, 1.0]))
#tasks[0].setGain(np.diag([5.0, 5.0]))
#tasks[0].setGain(np.diag([10.0, 10.0]))

# Simulation params
dt = 1.0/60.0
err_log = []
time_log = []
current_time = 0.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
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

    # Reset the desired position of the end effector every repeat
    for task in tasks:
        task_name = task.name
        if task_name == "End-effector position" or task_name == "End-effector configuration":
            new_x = np.random.uniform(0.0, 1.0)
            new_y = np.random.uniform(0.0, 1.0)
            desired = task.getDesired()
            desired[0][0] = new_x
            desired[1][0] = new_y
            task.setDesired(desired)
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global err_log, time_log, current_time, dt # variables for logging 

    ### Recursive Task-Priority algorithm
    # Initialize null-space projector
    dof = robot.getDOF() 
    P = np.eye(dof) 
    # Initialize output vector (joint velocity)
    dq = np.zeros((dof, 1))
    # Loop over tasks
    err_tasks = []
    for i in range(len(tasks)):
        # Update task state
        tasks[i].update(robot)
        
        # Retrieve the new K and v_ff parameters
        Ji = tasks[i].getJacobian()
        ei = tasks[i].getError()
        Ki = tasks[i].getGain()
        v_ff = tasks[i].getFeedforward()

        # Calculate task velocity
        sigma_dot = Ki @ ei + v_ff

        # Recursive Task-Priority
        Jbar_i = Ji @ P
        # Use DLS for matrix stabilization
        Jbar_dls = DLS(Jbar_i, 0.1)
        
        # Update dq using sigma_dot
        dq = dq + Jbar_dls @ (sigma_dot - Ji @ dq)
    
        # Update Null-space projector
        P = P - Jbar_dls @ Jbar_i
        
        # Store error data (norm)
        err_tasks.append(np.linalg.norm(ei))
    ###

    err_log.append(err_tasks)
    current_time += dt
    time_log.append(current_time)

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()


# After closing the simulation window, plot the control errors
err_log = np.array(err_log)  # Convert the error log list to a numpy array for easy indexing
plt.figure()

# Loop through each task to plot its error progression
for i in range(len(tasks)):
    # Plotting the norm of the error for task i over time
    plt.plot(time_log, err_log[:, i], label=tasks[i].name)

plt.xlabel('Time [s]')                # X-axis label
plt.ylabel('Control Error Norm')      # Y-axis label (magnitude of error)
plt.legend()                          # Show legend to identify different tasks
plt.title('Evolution of Task-Priority Control Errors') # Plot title
plt.grid(True)                        # Enable grid for better readability
plt.show()                            # Display the final plot