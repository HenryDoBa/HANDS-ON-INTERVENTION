import numpy as np
import matplotlib.pyplot as plt

# Load data
err_trans = np.load('error_transpose.npy')
err_pinv = np.load('error_pinv.npy')
err_dls = np.load('error_dls.npy')


dt = 1.0/60.0 
t_trans = np.arange(len(err_trans)) * dt
t_pinv = np.arange(len(err_pinv)) * dt
t_dls = np.arange(len(err_dls)) * dt

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_trans, err_trans, label='Transpose')
plt.plot(t_pinv, err_pinv, label='Pseudoinverse')
plt.plot(t_dls, err_dls, label='DLS')

plt.title('Evolution of the control error')
plt.xlabel('Time [s]')
plt.ylabel('Error [m]')
plt.legend()
plt.grid(True)
plt.show()