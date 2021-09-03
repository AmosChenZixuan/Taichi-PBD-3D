import taichi as ti
import numpy as np
from include.data import *

@ti.data_oriented
class Camera:
    SIN0 = ti.sin(0)
    COS0 = ti.cos(0)
    CENTER3 = vec3(.5, .5, .5)
    CENTER2 = vec2(.5, .5)

    def __init__(self, focus=CENTER3, angle=arr2(), scale=1.):
        self.angle = field((), 2, ti.f32)
        self.focus = field((), 3, ti.f32)
        self.scale = field((), 1, ti.f32)
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
        self.angle[None] += dp, dt
        
    def move(self, dx, dy, dz):
        self.focus[None] += dx, dy, dz

    def zoom(self, gamma):
        self.scale[None] *= gamma

    @ti.func
    def project(self, v3):
        p, t = self.getPhi(), self.getTheta()
        cp, sp = ti.cos(p), ti.sin(p)
        ct, st = ti.cos(t), ti.sin(t)

        x,y,z = v3 - self.getFocus()
        x, z = x * cp + z * sp, z * cp - x * sp
        u, v = x, y * ct + z * st
        return vec2(u,v) * self.getScale() + vec2(.5, .5)

    def toTexture(self, v3):
        cp, sp = self.COS0, self.SIN0
        ct, st = self.COS0, self.SIN0

        x,y,z = v3 - self.CENTER3
        x, z = x * cp + z * sp, z * cp - x * sp
        u, v = x, y * ct + z * st
        return vec2(u,v) + self.CENTER2


    

    