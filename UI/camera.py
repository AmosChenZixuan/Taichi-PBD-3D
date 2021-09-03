import taichi as ti
import numpy as np
from include.data import *

@ti.data_oriented
class Camera:
    def __init__(self, focus=arr3(), angle=arr2(), scale=1.):
        self.angle = field((), 2, ti.f32)
        self.focus = field((), 3, ti.f32)
        self.scale = field((), 1, ti.f32)
        self.init_state = (focus, angle, scale)
        self.reset()

    def reset(self):
        focus, angle, scale = self.init_state
        self.focus[None] = focus
        self.angle[None] = angle
        self.scale[None] = scale

    @ti.pyfunc
    def getPhi(self):
        return self.angle[None][0]* np.pi / 180.

    @ti.pyfunc
    def getTheta(self):
        return self.angle[None][1]* np.pi / 180.
    
    @ti.pyfunc
    def getFocus(self):
        return self.focus[None]

    @ti.pyfunc
    def getScale(self):
        return self.scale[None]

    def rotate(self, dp, dt):
        self.angle[None][0] += dp
        self.angle[None][1] += dt
        
    def move(self, dx, dy, dz):
        self.focus[None][0] += dx
        self.focus[None][1] += dy
        self.focus[None][2] += dz

    def zoom(self, gamma):
        self.scale[None] *= gamma

    @ti.pyfunc
    def project(self, v3):
        p, t = self.getPhi(), self.getTheta()
        cp, sp = ti.cos(p), ti.sin(p)
        ct, st = ti.cos(t), ti.sin(t)

        x = v3[0] - self.focus[None][0]
        y = v3[1] - self.focus[None][1]
        z = v3[2] - self.focus[None][2]
        x, z = x * cp + z * sp, z * cp - x * sp
        u, v = x, y * ct + z * st
        return vec2(u,v) * self.getScale() + vec2(.5, .5)


    

    