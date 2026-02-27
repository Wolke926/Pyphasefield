import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设数据
def solid_phase_energy(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

def liquid_phase_energy(x, y):
    return np.cos(np.sqrt(x**2 + y**2))

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z1 = solid_phase_energy(X, Y)
Z2 = liquid_phase_energy(X, Y)

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 画固相能量
ax.plot_surface(X, Y, Z1, color='b', alpha=0.7, label='Solid Phase Energy')

# 画液相能量
ax.plot_surface(X, Y, Z2, color='r', alpha=0.7, label='Liquid Phase Energy')

# 设置图形属性
ax.set_xlabel('X_ Al')
ax.set_ylabel('X_Nb')
ax.set_zlabel('Energy')
ax.set_title('Energy vs. Concentration at a Given Temperature')

plt.show()
