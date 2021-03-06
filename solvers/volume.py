'''
    Position-Based Simulation Methods in Computer Graphics
        http://mmacklin.com/EG2015PBD.pdf
'''
import taichi as ti
from include import *


@ti.data_oriented
class VolumeSolver:
    def __init__(self, memory: Memory, nParticles, nTets, retStf=1.):
        self.mem  = memory
        self.size = nTets
        self.retStf = retStf
        
        self.K    = field((), 1, ti.f32)        # stiffness (expansion)
        self.Tets = field(nTets, 4, ti.i32)     # vertices of tetrahedrons
        self.V0   = field(nTets, 1, ti.f32) # rest postion matrix inversed
        self.dp   = field(nParticles, 3, ti.f32)# postion delta  
        self.w    = field(nParticles, 1, ti.i32)# weights; number of springs on each vertex 
        
        self.NegK = 1.                          # stiffness (compression)

    def reset(self):
        self.K[None] = self.retStf
        self.w.fill(0)

    def update(self, i, x, y, z, w):
        self.Tets[i] = x,y,z,w
        self.w[x] += 1#4**2
        self.w[y] += 1#4**2
        self.w[z] += 1#4**2
        self.w[w] += 1#4**2

    def init(self):
        self.initRestVolume()


    #@timeThis
    def solve(self):
        self.clearDelta()
        self.calcDelta()
        self.applyDelta()

    ################### Private Methods #####################

    @ti.kernel
    def initRestVolume(self):
        mem = self.mem
        for i in range(self.size):
            x, y, z, w  = self.Tets[i]
            px,py,pz,pw = mem.curPos[x], mem.curPos[y], mem.curPos[z], mem.curPos[w]  
            self.V0[i]  = 1/6. * abs(
                (pw-px).dot((pz-px).cross(py-px))
            )

    def clearDelta(self):
        self.dp.fill(0)

    @ti.kernel
    def calcDelta(self):
        mem = self.mem
        eps = 1e-9
        for i in range(self.size):
            x, y, z, w  = self.Tets[i]
            px,py,pz,pw = mem.newPos[x], mem.newPos[y], mem.newPos[z], mem.newPos[w]  
            v           = 1/6. * (py-px).cross(pz-px).dot(pw-px)
            if (self.K[None]==0. and v > 0.): continue
            if (self.NegK   ==0. and v < 0.): continue

            d0 = (py-pz).cross(pw-pz)
            d1 = (pz-px).cross(pw-px)
            d2 = (px-py).cross(pw-py)
            d3 = (py-px).cross(pz-px)
            gradSum = d0.norm_sqr()*mem.invM[x] + d1.norm_sqr()*mem.invM[y] + \
                    d2.norm_sqr()*mem.invM[z] + d3.norm_sqr()*mem.invM[w]
            if abs(gradSum) > eps: 
                #k       = self.NegK if v < 0. else self.K[None]
                k = self.K[None]
                vlambda = k * (v-self.V0[i]) / gradSum

                self.dp[x] -= vlambda * mem.invM[x] * d0
                self.dp[y] -= vlambda * mem.invM[y] * d1
                self.dp[z] -= vlambda * mem.invM[z] * d2
                self.dp[w] -= vlambda * mem.invM[w] * d3

    @ti.kernel
    def applyDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z,w  = self.Tets[i]
            mem.newPos[x] += self.dp[x] / self.w[x]
            mem.newPos[y] += self.dp[y] / self.w[y]
            mem.newPos[z] += self.dp[z] / self.w[z]
            mem.newPos[w] += self.dp[w] / self.w[w]