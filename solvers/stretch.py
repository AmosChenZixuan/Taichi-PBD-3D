import taichi as ti
import numpy as np

from include import *

@ti.data_oriented
class TotalStretchSolver:
    def __init__(self, memory:Memory, nParticles, nSprings):
        self.mem  = memory
        self.size = nSprings

        self.K       = field((), 1, ti.f32)         # stiffness
        self.Springs = field(nSprings, 2, ti.i32)   # vertices of springs
        self.restLen = field(nSprings, 1, ti.f32)   # rest length
        self.dp   = field(nParticles, 3, ti.f32)    # postion delta  
        self.w    = field(nParticles, 1, ti.i32)    # weights; number of springs on each vertex


    def reset(self):
        self.K[None] = 1.
        self.w.fill(0) 

    def update(self, i, x, y):
        self.Springs[i] = x,y
        self.w[x] += 2**3
        self.w[y] += 2**3

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
            x,y = self.Springs[i]
            r = mem.curPos[x] - mem.curPos[y]
            self.restLen[i] = r.norm()

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
                dp = n.normalized() * (d-self.restLen[i]) / (w1 + w2)
                self.dp[x] -= dp * w1 * self.K[None] 
                self.dp[y] += dp * w2 * self.K[None] 

    @ti.kernel
    def applyDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y = self.Springs[i]
            mem.newPos[x] += self.dp[x] / self.w[x]
            mem.newPos[y] += self.dp[y] / self.w[y]
            #print(self.dp[x], self.dp[y], self.w[x], self.w[y])

