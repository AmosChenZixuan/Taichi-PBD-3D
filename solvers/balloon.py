'''
    http://mmacklin.com/EG2015PBD.pdf
'''
import taichi as ti
from include import *


@ti.data_oriented
class ClothBalloonSolver:
    def __init__(self, memory: Memory, nParticles, nTris, rstP=.1):
        '''
        @params
            memory:     object that stores all info of particles
            nParticles: particle number of the balloon
            nTris     : triangle number of the ballon
            rstP      : initial pressure factor
        '''
        self.mem  = memory
        self.size = nParticles, nTris
        self.rstP = rstP
        self.ptr  = field(nParticles, 1, ti.i32) # device particle index
        self.points = set()                      # host   particle index, init helper                  

        self.K    = field((), 1, ti.f32)         # pressure
        self.Tris = field(nTris, 3, ti.i32)      # vertices of triangles
        self.Vol  = field(2, 1, ti.f32)          # [rest volume, cur volume]
        self.grad = field(memory.size, 3, ti.f32)# constraint gradients 

    def reset(self):
        self.points  = set() 
        self.K[None] = self.rstP
        self.Vol.fill(0.)

    def update(self, i, x, y, z):
        self.Tris[i] = x,y,z
        for idx in [x,y,z]:
            if idx not in self.points:
                self.ptr[len(self.points)] = idx
                self.points.add(idx)


    def init(self):
        assert len(self.points) == self.size[0]
        self.clearCache()
        self.updateVolume()
        self.Vol[0] = self.Vol[1]   # assign rest volume

    #@timeThis
    def solve(self):
        self.clearCache()
        self.updateVolume()
        self.updateGradients()
        self.project()

    ################### Private Methods #####################

    def clearCache(self):
        self.grad.fill(0.)

    @ti.kernel
    def updateVolume(self):
        '''
            Sum_tri{ cross(x,y) dot z}
        '''
        mem = self.mem
        self.Vol[1] = 0.
        for i in range(self.size[1]):
            x, y, z     = self.Tris[i]
            px,py,pz    = mem.newPos[x], mem.newPos[y], mem.newPos[z]
            self.Vol[1] += px.cross(py).dot(pz)
        #print(self.Vol[1])

    @ti.kernel
    def updateGradients(self):
        mem = self.mem
        for i in range(self.size[1]):
            x, y, z  = self.Tris[i]
            px,py,pz = mem.newPos[x], mem.newPos[y], mem.newPos[z]
            self.grad[x] += py.cross(pz)
            self.grad[y] += pz.cross(px)
            self.grad[z] += px.cross(py)

    @ti.kernel
    def project(self):
        mem = self.mem
        gradSum = 0.
        Cx      = self.Vol[1] - self.Vol[0]*self.K[None]
        for i in range(self.size[0]):
            x = self.ptr[i]
            gradSum += self.grad[x].norm_sqr() * mem.invM[x]
        for i in range(self.size[0]):
            if abs(gradSum) < 1e-9:
                print('WTF')
                break
            x = self.ptr[i]
            mem.newPos[x] += -(Cx / gradSum) * mem.invM[x] * self.grad[x]

    
