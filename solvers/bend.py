import taichi as ti
from include import *


@ti.data_oriented
class TotalBendingSolver:
    def __init__(self, memory:Memory, nParticles, nTris, restStiff=1.):
        self.mem  = memory
        self.size = nTris
        self.reStf= restStiff
        
        self.K       = field((), 1, ti.f32)        # stiffness
        self.Tris    = field(nTris, 3, ti.i32)     # vertices of triangles
        self.restLen = field(nTris, 1, ti.f32)     # rest length
        self.dp      = field(nParticles, 3, ti.f32)# postion delta  
        self.w       = field(nParticles, 1, ti.i32)# weights; number of springs on each vertex 

    def reset(self):
        self.K[None] = self.reStf
        self.w.fill(0)

    def update(self, i, x, y, z):
        self.Tris[i] = x,y,z
        self.w[x] += 1
        self.w[y] += 1
        self.w[z] += 1

    def init(self):
        self.initRestLen()

    #@timeThis
    def solve(self):
        self.clearDelta()
        self.calcDelta()
        self.applyDelta()

    ################### Private Methods #####################

    @ti.kernel
    def initRestLen(self):
        mem = self.mem
        for i in range(self.size):
            x, y, z  = self.Tris[i]
            trg = mem.curPos[z]
            cm  = (mem.curPos[x] + mem.curPos[y] + trg) / 3.
            self.restLen[i] = (trg - cm).norm()
        
        # for i in self.w:
        #     self.w[i] = self.w[i]**2

    def clearDelta(self):
        self.dp.fill(0)

    @ti.kernel
    def calcDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z  = self.Tris[i]
            trg    = mem.newPos[z]
            cm     = (mem.newPos[x] + mem.newPos[y] + trg) / 3.
            diff   = trg - cm
            dist   = diff.norm()
            if dist > 0.:
                C  = 1. - (self.restLen[i] / dist)
                dp = diff * C * self.K[None]
                self.dp[x] += dp * 2 / 4
                self.dp[y] += dp * 2 / 4
                self.dp[z] -= dp * 4 / 4
            else:
                print("wwwwww")



    @ti.kernel
    def applyDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z  = self.Tris[i]
            mem.newPos[x] += self.dp[x] / self.w[x]
            mem.newPos[y] += self.dp[y] / self.w[y]
            mem.newPos[z] += self.dp[z] / self.w[z]
            #print(self.dp[x], self.dp[y], self.dp[z])