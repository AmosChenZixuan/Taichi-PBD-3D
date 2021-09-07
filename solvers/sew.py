import taichi as ti
import numpy as np

from include import *

@ti.data_oriented
class TotalSewSolver:
    def __init__(self, memory:Memory, nParticles, nSprings, restStiff=1., restLen=0.):
        self.mem  = memory
        self.size = nSprings
        self.reStf= restStiff

        self.K       = field((), 1, ti.f32)         # stiffness
        self.Springs = field(nSprings, 2, ti.i32)   # vertices of springs
        self.dp   = field(nParticles, 3, ti.f32)    # postion delta  
        self.w    = field(nParticles, 1, ti.i32)    # weights; number of springs on each vertex
        self.restLen = restLen


    def reset(self):
        self.K[None] = self.reStf
        self.w.fill(0) 

    def update(self, i, x, y):
        self.Springs[i] = x,y
        self.w[x] += 1
        self.w[y] += 1

    def init(self):
        self.initRelaxation()

    #@timeThis
    def solve(self):
        self.clearDelta()
        self.calcDelta()
        self.applyDelta()

    ################### Private Methods #####################

    @ti.kernel
    def initRelaxation(self):
        for i in self.w:
            self.w[i] = self.w[i]**2

    def clearDelta(self):
        self.dp.fill(0)

    @ti.kernel
    def calcDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y = self.Springs[i]
            w1, w2 = mem.invM[x], mem.invM[y]
            n  = mem.newPos[x] - mem.newPos[y] + self.dp[x] - self.dp[y]
            d  = n.norm()
            if w1 + w2 > 0. and d > 0.:
                dp = n.normalized() * (d-self.restLen) / (w1 + w2)
                self.dp[x] -= dp * w1 * self.K[None] 
                self.dp[y] += dp * w2 * self.K[None] 

    @ti.kernel
    def applyDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y = self.Springs[i]
            mem.newPos[x] += self.dp[x] / self.w[x]
            mem.newPos[y] += self.dp[y] / self.w[y]
            #print('s', self.dp[x], self.dp[y], self.w[x], self.w[y])

