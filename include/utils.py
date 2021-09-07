
import time
import numpy as np
from numpy.linalg import norm
from include.data import vec2, vec3

def timeThis(func):
    def wapper(*args, **kargs):
        s = time.time()
        result = func(*args, **kargs)
        print(f'{func.__name__!r} executed in {(time.time()-s):.4f}s')
        return result
    return wapper

def pick(pos, mouse_pos):
    '''
     Among all points in pos, pick the closest to mouse_pos
    '''
    pos = (pos - mouse_pos) * -1              # mouse-pos = -(pos-mouse)
    dists = np.array([norm(v) for v in pos])
    closest = int(np.argmin(dists))
    return closest if dists[closest] < 0.1 else -1

def flatten(v3):
    cp, sp = np.cos(0), np.sin(0)
    ct, st = np.cos(0), np.sin(0)

    x,y,z = vec3(*v3)-.5
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return vec2(u,v) * 1. +.5


# @ti.pyfunc
# def tri_norm(i1,i2,i3):
#     p1, p2, p3 = P[i1], P[i2], P[i3]
#     u = vec3(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
#     v = vec3(p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
#     return u.cross(v)
#     # nx = u[1]*v[2] - u[2]*v[1]
#     # ny = u[2]*v[0] - u[0]*v[2]
#     # nz = u[0]*v[1] - u[1]*v[0]
#     # return vec3(nx, ny, nz)