import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Your existing code for loading and calculations...
xyz = pd.read_csv("/home/ppacaud/xyz.csv")
xyz = xyz.to_numpy()
point_of_interest = [0.27847117, -0.25263312, 1.146976]
scale_factor = 0.25

x_extent = xyz[:, 0].max() - xyz[:, 0].min()
y_extent = xyz[:, 1].max() - xyz[:, 1].min()
z_extent = xyz[:, 2].max() - xyz[:, 2].min()

# Calculate bounds to keep 1/4th of the actual point cloud extent
x_min = point_of_interest[0] - x_extent * scale_factor / 2
x_max = point_of_interest[0] + x_extent * scale_factor / 2
y_min = point_of_interest[1] - y_extent * scale_factor / 2
y_max = point_of_interest[1] + y_extent * scale_factor / 2
z_min = point_of_interest[2] - z_extent * scale_factor / 2
z_max = point_of_interest[2] + z_extent * scale_factor / 2

mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
       (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
       (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)

print(mask)
print(xyz[mask])

# scatter plot 3d xyz
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
plt.show()
