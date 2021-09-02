import numpy as np


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