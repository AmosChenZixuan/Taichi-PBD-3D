import taichi as ti
from include.data import *

@ti.data_oriented
class Memory:
    def __init__(self, nParticles):
        self.size   = nParticles
        self.curPos = field(nParticles, 3, ti.f32)
        self.newPos = field(nParticles, 3, ti.f32)
        self.vel    = field(nParticles, 3, ti.f32)
        self.invM   = field(nParticles, 1, ti.f32)

    def __len__(self):
        return self.size

    def update(self, i, pos, mass=1.):
        self.curPos[i] = pos
        self.newPos[i] = pos
        self.vel[i]    = vec3()
        self.invM[i]   = mass

    