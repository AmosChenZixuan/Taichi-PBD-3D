from UI.camera import Camera
from include.memory import Memory
import taichi as ti
import numpy as np
from numpy.linalg import norm
from include.data import *

@ti.data_oriented
class PostionBasedDynamics:
    def __init__(self, memory:Memory, camera:Camera, nParticle):
        self.mem  = memory
        self.cam  = camera
        self.size = nParticle

        self.gravity = field((), 3, ti.f32)
        self.iters   = field((), 1, ti.i32)
        self.substep = 2 
        self.dt      = 1 / 60 / self.substep 

    def reset(self):
        self.gravity[None] = vec3(y=-9.8)
        self.iters[None]   = 3
        self.substep = 2

    def init(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    @ti.kernel
    def apply_force(self, mouse_x: ti.f32, mouse_y: ti.f32, idx: ti.i32):
        mem = self.mem
        for i in range(self.size):
            if mem.invM[i] <= 0.: continue
            mem.vel[i] += self.gravity[None] * self.dt
            mem.newPos[i]  = mem.curPos[i] + mem.vel[i] * self.dt

            # mouse interaction
            if idx>=0:
                mem.newPos[idx] = mouse_x, mouse_y, mem.newPos[idx][2]

    @ti.kernel
    def update(self):
        mem = self.mem
        for i in range(self.size):
            if mem.invM[i] <= 0.:
                mem.newPos[i] = mem.curPos[i]
            else:
                mem.vel[i] = (mem.newPos[i] - mem.curPos[i]) / self.dt * .99
                mem.curPos[i] = mem.newPos[i]

    @ti.kernel
    def box_confinement(self):
        mem = self.mem
        for i in range(self.size):
            if mem.newPos[i][1] < 0.:
                mem.newPos[i][1] = 1e-4

    @ti.kernel
    def project(self, p3: ti.ext_arr(), p2: ti.ext_arr()):
        '''
            Input  p3: np.ndarray, 3d positions
            Output p2: np.ndarray, 2d positions
        '''
        for i in range(p3.shape[0]):
            x,y,z = p3[i,0], p3[i,1], p3[i,2]
            u,v   = self.cam.project(vec3(x,y,z))
            p2[i,0] = u
            p2[i,1] = v