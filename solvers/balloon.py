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
        self.dp   = field(memory.size, 3, ti.f32)# postion delta  
        self.w    = field(memory.size, 1, ti.i32)# weights; number of springs on each vertex  

    def reset(self):
        self.points  = set() 
        self.K[None] = self.rstP
        self.w.fill(0)  
        self.Vol.fill(0.)

    def update(self, i, x, y, z):
        self.Tris[i] = x,y,z
        for idx in [x,y,z]:
            if idx not in self.points:
                self.ptr[len(self.points)] = idx
                self.points.add(idx)
        self.w[x] += 16
        self.w[y] += 16
        self.w[z] += 16


    def init(self):
        self.clearDelta()
        self.updateVolume()
        self.Vol[0] = self.Vol[1]   # assign rest volume

    #@timeThis
    def solve(self):
        self.clearDelta()
        self.updateVolume()
        self.updateGradients()
        self.calcDelta()
        self.applyDelta()

    ################### Private Methods #####################

    def clearDelta(self):
        self.dp.fill(0.) 
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
    def calcDelta(self):
        mem = self.mem
        for i in range(self.size[1]):
            x, y, z  = self.Tris[i]
            Cx       = self.Vol[1] - self.Vol[0]*self.K[None]
            dx,dy,dz = self.grad[x], self.grad[y], self.grad[z]

            gradSum = dx.norm_sqr() * mem.invM[x] + dy.norm_sqr() * mem.invM[y] + \
                        dz.norm_sqr() * mem.invM[z]
            if abs(gradSum) > 1e-9:
                vlambda = Cx / gradSum
                self.dp[x] -= vlambda * mem.invM[x] * dx
                self.dp[y] -= vlambda * mem.invM[y] * dy
                self.dp[z] -= vlambda * mem.invM[z] * dz
            else:
                print('WTF')

    @ti.kernel
    def applyDelta(self):
        mem = self.mem
        for i in range(self.size[0]):
            x = self.ptr[i]
            mem.newPos[x] += self.dp[x] / self.w[x]**2

    
