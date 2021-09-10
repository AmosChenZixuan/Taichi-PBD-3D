import numpy as np
import pygalmesh

def createSphere(alpha, beta, radius=1, pos=(0.,0.,0.)):
    vertices = []
    deltaTheta =   np.pi/alpha
    deltaPhi   = 2*np.pi/beta
    theta,phi  = 0,0
    # where 0 <= theta < 2*pi
    for ring in range(alpha-1):
        theta += deltaTheta
        for p in range(beta):
            phi += deltaPhi
            x = np.sin(theta) * np.cos(phi) * radius + pos[0]
            y = np.sin(theta) * np.sin(phi) * radius + pos[1]
            z = np.cos(theta) * radius + pos[2]
            vertices.append([x,y,z])
    # add two poles
    vertices.append([pos[0],pos[1], 1*radius + pos[2]])
    vertices.append([pos[0],pos[1],-1*radius + pos[2]])
    return vertices

    
def createRandomCloth(n, center=(0.,0.,0.)):
    x = .5 * np.random.rand(n) + center[0]/2
    y = .5 * np.random.rand(n) + center[1]/2
    z = 1e-5 * np.random.rand(n)+center[2]
    return list(np.vstack([x, y, z]).T)

def createRectCloth(w, h, dw, dh, leftbtm=(0.,0.,0.)):
    points = []
    for i in range(w):
        for j in range(h):
            x = dw * i + leftbtm[0]
            y = dh * j + leftbtm[1]
            z = 1e-9*i*j + leftbtm[2]
            points.append([x,y,z])
    return points

def createBowCloth(w, h, dw, dh, radius=.1, leftbtm=(0.,0.,0.), dir=1.):
    points = []
    for i in range(w):
        for j in range(h):
            x = leftbtm[0] + dw * i
            y = leftbtm[1] + dh * j
            z = leftbtm[2] + ( np.sin(np.pi*(i//w + (j+.5)/h)) + np.sin(np.pi*(j//h + (i+.5)/w)) ) * radius/2 * dir
            points.append([x,y,z])
    return points

def createIDK(w, h, dw, dh, leftbtm=(0.,0.,0.)):
    points = []
    for i in range(w):
        for j in range(h):
            x = leftbtm[0] + dw * i
            y = leftbtm[1] + dh * j
            z = leftbtm[2] + ( np.sin(np.pi*(i//w + (j+.5)/h)) + np.sin(np.pi*(j//h + (i+.5)/w)) ) /2 * .1
            points.append([x,y,z])
            z = 1e-9*i*j + leftbtm[2]
            points.append([x,y,z])
    return points

def createGalCube(w,h,d, res):
    x_ = np.linspace(.0, w, res)
    y_ = np.linspace(.0, h, res)
    z_ = np.linspace(.0, d, res)
    x, y, z = np.meshgrid(x_, y_, z_)

    vol = np.empty((res, res, res), dtype=np.uint8)
    idx = abs(x) + abs(y) + abs(z) < 1.5
    vol[idx] = 1
    vol[~idx] = 0
    voxel_size = (0.1, 0.1, 0.1)

    mesh = pygalmesh.generate_from_array(
        vol, voxel_size, max_facet_distance=0.2, max_cell_circumradius=0.5, verbose=False
    )
    return np.array(mesh.points)/10 + 0.5, mesh.cells

def createGalBall(r=1.):
    s = pygalmesh.Ball([0, 0, 0], r)
    mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=0.5, verbose=False)
    return np.array(mesh.points)/10 + 0.5, mesh.cells

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay

    points = np.array(createBowCloth(50,25,0.02, 0.02, .1,(.25, .25, .5)) + createRectCloth(50,25,0.02, 0.02, (.25, .25, .5)))

    tri = Delaunay(points)
    ig = plt.figure()
    ax = plt.axes(projection='3d')
    # for tr in tri.simplices:
    #     pts = points[tr, :]
    #     ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
    #     ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
    #     ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
    #     ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
    #     ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
    #     ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')

    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')
    plt.show()