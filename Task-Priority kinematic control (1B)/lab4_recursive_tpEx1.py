from lab4_roboticsEx1 import * # Includes numpy import
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
# a. One task -> 1: end-effector position
#tasks = [ Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1))] 

# b. One task -> 1: end-effector configuration  
#tasks = [Configuration2D("End-effector configuration", np.array([1.2, 0.8, 0.5]).reshape(3,1))]

# c. Two tasks -> 1: end-effector position, 2: end-effector orientation
#tasks = [
#   Position2D("End-effector position", np.array([1.2, 0.8]).reshape(2,1)),
#   Orientation2D("End-effector orientation", np.array([0.5]).reshape(1,1))
#]

# Two tasks -> 1: end-effector position, 2: joint 1 position
tasks = [
   Position2D("End-effector position", np.array([1.2, 0.8]).reshape(2,1)),
   JointPosition("Joint 1 position", np.array([0.0]).reshape(1,1), 0), # Joint 1 position task with desired value of 0.0 radians
]

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

        # Compute augmented Jacobian
        Ji = tasks[i].getJacobian()
        Jbar_i = Ji @ P

        # Compute task velocity
        Jbar_dls = DLS(Jbar_i, 0.1)
        err_i = tasks[i].getError()
        dq = dq + Jbar_dls @ (err_i - Ji @ dq)
    
        # Accumulate velocity
        # Update null-space projector
        P = P - np.linalg.pinv(Jbar_i) @ Jbar_i

        # Extract task name for conditional error logging
        task_name = tasks[i].name.lower()
        
        if "position" in task_name and "configuration" not in task_name:
            # For 2D Position tasks, log the Euclidean norm (magnitude) of the error
            err_tasks.append(np.linalg.norm(err_i))
            
        elif "orientation" in task_name:
            # For 2D Orientation, the error is typically a single scalar (rotation angle)
            err_tasks.append(err_i.flatten()[0]) 
            
        elif "configuration" in task_name:
            # Configuration tasks combine position and orientation
            # Extract the norm of the first two elements (x, y position error)
            pos_err = np.linalg.norm(err_i[:2])
            # Extract the third element (theta orientation error)
            ori_err = abs(err_i[2][0])
            
            # Store both separately to allow plotting two distinct lines
            err_tasks.append(pos_err)
            err_tasks.append(ori_err)
            
        else: 
            # Default case for JointPosition or other scalar-based tasks
            # Extracts the scalar value from the error array
            err_tasks.append(err_i.flatten()[0])
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
labels = [task.name for task in tasks]

fig2, ax_q = plt.subplots()

# Special handling for Configuration2D task (usually has 2 error components: pos and ori)
if len(tasks) == 1 and "configuration" in tasks[0].name.lower():
    # If the log has 2 columns for a single configuration task
    ax_q.plot(time_log, err_log[:, 0], label='End-effector position')
    ax_q.plot(time_log, err_log[:, 1], label='End-effector orientation')
else:
    # Standard plotting for multiple independent tasks
    # The number of columns in err_log corresponds to the number of tracked errors
    for i in range(err_log.shape[1]):
        # Check if we have a corresponding label, otherwise use the last known label
        task_label = labels[i] if i < len(labels) else labels[-1]
        # Plot using the format 'e{priority_number} {Task Name}'
        ax_q.plot(time_log, err_log[:, i], label=f'{task_label}')

# Set plot metadata to match the reference style
ax_q.set_title('Task-Priority Control Errors')
ax_q.set_xlabel('Time [s]')
ax_q.set_ylabel('Error Value')
ax_q.legend()
ax_q.grid(True)

plt.show() # Display the final plot