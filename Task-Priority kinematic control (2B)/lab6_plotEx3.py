from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt

plt.figure()
colors = {'Method_A': 'r', 'Method_B': 'g', 'Method_C': 'b'}

for m in ['Method_A', 'Method_B', 'Method_C']:
    try:
        bx = np.load(f'{m}_base_x.npy')
        by = np.load(f'{m}_base_y.npy')
        ex = np.load(f'{m}_ee_x.npy')
        ey = np.load(f'{m}_ee_y.npy')
        
        plt.plot(bx, by, label=f'Base {m}', color=colors[m], linestyle='--')
        plt.plot(ex, ey, label=f'EE {m}', color=colors[m])
    except:
        print(f"Chưa có dữ liệu cho {m}")

plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Comparison of Integration Methods')
plt.legend()
plt.grid()
plt.show()