# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

# Robot definition (3 revolute joint planar manipulator)
d = np.array([0, 0, 0])                             # displacement along Z-axis
q = np.array([0.2, 0.5, 0.2]).reshape(3,1)          # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.4])                      # displacement along X-axis
alpha = np.array([0, 0, 0])                         # rotation around X-axis 
revolute = [True, True, True]                       # flags specifying the type of joints

# Control Parameters
K1 = np.diag([2.0])                            # Gain cho Task 1 (Khớp 1) - Ưu tiên cao
K2 = np.diag([2.0, 2.0])                       # Gain cho Task 2 (End-effector) - Ưu tiên thấp
damping = 0.1                                  # Damping factor cho DLS
dq_max = 1.0                                   # Vận tốc khớp tối đa

# Desired values of task variables
sigma1_d = np.array([[0.0]])                   # Mục tiêu Task 1: Khớp 1 giữ ở góc 0
sigma2_d = np.array([0.0, 1.0]).reshape(2,1)   # Mục tiêu Task 2: Vị trí EE

# Simulation params
dt = 1.0/60.0
Tt = 10 
tt = np.arange(0, Tt, dt)

# Save history
err_ee_history = []
joint_1_record = []
time_history = []

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Task-Priority Control (Improved)')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-b', lw=2, label='Robot') 
path, = ax.plot([], [], 'c--', lw=1, label='Path') 
point, = ax.plot([], [], 'rx', ms=10, label='Target')
ax.legend()
PPx, PPy = [], []

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
    
    # Cập nhật mục tiêu ngẫu nhiên cho EE mỗi chu kỳ 10s (tùy chọn)
    if t <= dt:
        sigma2_d = np.random.uniform(-1.0, 1.0, (2, 1))

    # 1. Kinematics & Jacobian
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    # --- TASK 1: Điều khiển khớp 1 (Ưu tiên cao nhất) ---
    sigma1 = np.array([[q[0,0]]])               # Góc hiện tại khớp 1
    err1 = sigma1_d - sigma1                    # Sai số Task 1
    J1 = np.array([[1.0, 0.0, 0.0]])            # Jacobian của Task 1
    
    J1_star = DLS(J1, damping)                  # Nghịch đảo DLS
    dq1 = J1_star @ (K1 @ err1)                 # Vận tốc khớp cho Task 1
    
    # Tính Null Space Projector của Task 1 (Sử dụng pinv để chính xác tuyệt đối)
    P1 = np.eye(3) - np.linalg.pinv(J1) @ J1

    # --- TASK 2: Điều khiển vị trí EE (Ưu tiên thấp hơn) ---
    sigma2 = T[-1][0:2, 3].reshape(2, 1)        # Vị trí EE hiện tại
    err2 = sigma2_d - sigma2                    # Sai số Task 2
    J2 = J[0:2, :]                              # Jacobian của Task 2 (x, y)
    
    # Chiếu Task 2 vào không gian vô hiệu của Task 1
    J2_aug = J2 @ P1 
    J2_aug_star = DLS(J2_aug, damping)          # Nghịch đảo DLS cho ma trận đã chiếu
    
    # 2. Kết hợp các Task (Task Priority Formula)
    # dq = dq1 + J2_aug# * (v2 - J2 * dq1)
    dq_total = dq1 + J2_aug_star @ (K2 @ err2 - J2 @ dq1)

    # 3. Giới hạn vận tốc (Scaling)
    s = np.max(np.abs(dq_total) / dq_max)
    dq_final = dq_total / s if s > 1 else dq_total

    # 4. Cập nhật trạng thái
    q = q + dq_final * dt 

    # Cập nhật hình ảnh
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma2_d[0], sigma2_d[1])

    # Lưu dữ liệu vẽ biểu đồ
    err_ee_history.append(np.linalg.norm(err2))
    joint_1_record.append(q[0, 0])
    time_history.append(t)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, frames=np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()

# Biểu đồ sai số
plt.figure(figsize=(10, 5))
plt.subplot(2,1,1)
plt.plot(time_history, err_ee_history, 'r', label='EE Position Error')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(time_history, joint_1_record, 'b', label='Joint 1 Angle')
plt.axhline(y=sigma1_d[0,0], color='k', linestyle='--', label='Target q1')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()