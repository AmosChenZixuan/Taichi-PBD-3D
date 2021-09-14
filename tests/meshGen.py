import pygalmesh
import numpy as np
import matplotlib.pyplot as plt
from include import *

# points = []
# points.extend(createBowCloth(25,25,0.02, 0.02, .1, (.25, .25, .5)))
# points.extend(createRectCloth(25,25,0.02, 0.02,(.25, .25, .5)))

x_ = np.linspace(-1.0, 1.0, 50)
y_ = np.linspace(-1.0, 1.0, 50)
z_ = np.linspace(-1.0, 1.0, 50)
x, y, z = np.meshgrid(x_, y_, z_)

vol = np.empty((50, 50, 50), dtype=np.uint8)
idx = abs(x) + abs(y) + abs(z) < 5
vol[idx] = 1
vol[~idx] = 0

voxel_size = (0.1, 0.1, 0.1)

mesh = pygalmesh.generate_from_array(
    vol, voxel_size, max_facet_distance=0.2, max_cell_circumradius=0.5
)


fig = plt.figure()
ax = plt.axes(projection='3d')

points = mesh.points
ax.scatter(points[:,0], points[:,1], points[:,2], color='b')
plt.show()
t=0