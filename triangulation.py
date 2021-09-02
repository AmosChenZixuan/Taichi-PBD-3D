import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def plot_tri(ax, points, tri):
    for tr in tri.simplices:
        pts = points[tr, :]
        ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
        ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
        ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
        ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')

    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')


def tex(p3d):
    cp, sp = np.cos(0), np.sin(0)
    ct, st = np.cos(0), np.sin(0)

    x,y,z = p3d - [0.5, 0.5, 0.5]
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return [u,v] * 1. + 0.5

# np.random.seed(0)
# x = 2.0 * np.random.rand(10) - 1.0
# y = 2.0 * np.random.rand(10) - 1.0
# z = 2.0 * np.random.rand(10) - 1.0
# points = np.vstack([x, y, z]).T
points = [[.5,.5,.5]]
deltaTheta = np.pi/10
deltaPhi =   np.pi/5
theta,phi = 0,0
for ring in range(10):
    theta += deltaTheta
    for p in range(10):
        phi += deltaPhi
        x = np.sin(theta) * np.cos(phi) / 5 + 0.5
        y = np.sin(theta) * np.sin(phi) / 5 + 0.5
        z = np.cos(theta) / 5 + 0.5
        points.append([x,y,z])

for ring in range(10):
    theta += deltaTheta
    for p in range(10):
        phi += deltaPhi
        x = np.sin(theta) * np.cos(phi) / 10 + 0.5
        y = np.sin(theta) * np.sin(phi) / 10 + 0.5
        z = np.cos(theta) / 10 + 0.5
        points.append([x,y,z])
points = np.array(points)
tri = Delaunay(points)

fig = plt.figure()
ax = plt.axes(projection='3d')
plot_tri(ax, points, tri)
plt.show()
t=0