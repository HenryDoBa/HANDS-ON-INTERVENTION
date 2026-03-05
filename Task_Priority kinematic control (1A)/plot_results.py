import numpy as np
import matplotlib.pyplot as plt

# 1. Load the saved data from the simulation runs
try:
    err1_A = np.load('error1_caseA.npy')  # Task 1 error (End-effector) for Case A
    err2_A = np.load('error2_caseA.npy')  # Task 2 error (Joint 1) for Case A
    err1_B = np.load('error1_caseB.npy')  # Task 1 error (End-effector) for Case B
    err2_B = np.load('error2_caseB.npy')  # Task 2 error (Joint 1) for Case B
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# 2. Define time parameters
dt = 1.0/60.0  # Simulation step (60Hz)

# Create separate time vectors for Case A and Case B to avoid dimension mismatch
# Case A and Case B might have different total durations (lengths)
t_A = np.arange(len(err1_A)) * dt
t_B = np.arange(len(err1_B)) * dt

# 3. Initialize the figure with 2 subplots (stacked vertically)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.tight_layout(pad=5.0) # Add padding to prevent overlapping labels

# Subplot 1: Task 1 Error (End-Effector Position)
# plot err1_A against t_A and err1_B against t_B
ax1.plot(t_A, err1_A, label='Case A (EE Priority)')
ax1.plot(t_B, err1_B, label='Case B (Joint 1 Priority)')
ax1.set_title('Task 1 Error Comparison: End-Effector Position')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Error Norm [m]')
ax1.legend()
ax1.grid(True)

# Subplot 2: Task 2 Error (Joint 1 Position)
# plot err2_A against t_A and err2_B against t_B
ax2.plot(t_A, err2_A, label='Case A (EE Priority)')
ax2.plot(t_B, err2_B, label='Case B (Joint 1 Priority)')
ax2.set_title('Task 2 Error Comparison: Joint 1 Position')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Error Norm [rad]')
ax2.legend()
ax2.grid(True)

# 4. Display the finalized comparison plots
plt.show()