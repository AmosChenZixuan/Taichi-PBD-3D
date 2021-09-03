
import time

def timeThis(func):
    def wapper(*args, **kargs):
        s = time.time()
        result = func(*args, **kargs)
        print(f'{func.__name__!r} executed in {(time.time()-s):.4f}s')
        return result
    return wapper


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