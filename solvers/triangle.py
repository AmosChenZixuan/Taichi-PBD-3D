import taichi as ti
from include import *


@ti.data_oriented
class TriangleSolver:
    def __init__(self, memory: Memory, nParticles, nTris, restStiff=1.):
        self.mem  = memory
        self.size = nTris
        self.reStf= restStiff
        
        self.K    = field((), 1, ti.f32)        # stiffness
        self.Tris = field(nTris, 3, ti.i32)     # vertices of triangles
        self.invQ = field(nTris, (2,2), ti.f32) # rest postion matrix inversed
        self.dp   = field(nParticles, 3, ti.f32)# postion delta  
        self.w    = field(nParticles, 1, ti.i32)# weights; number of springs on each vertex 
        

    def reset(self):
        self.K[None] = self.reStf
        self.w.fill(0)

    def update(self, i, x, y, z):
        self.Tris[i] = x,y,z
        self.w[x] += 1
        self.w[y] += 1
        self.w[z] += 1


    def init(self):
        self.initInvQ()

    #@timeThis
    def solve(self):
        self.clearDelta()
        self.calcDelta()
        self.applyDelta()

    ################### Private Methods #####################

    @ti.kernel
    def initInvQ(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z = self.Tris[i]
            col0 = flatten(mem.curPos[y] - mem.curPos[x])
            col1 = flatten(mem.curPos[z] - mem.curPos[x])

            self.invQ[i]= mat2(col0, col1, byCol=True)
            self.invQ[i] = self.invQ[i].inverse()

        for i in self.w:
            self.w[i] = self.w[i]**2

    def clearDelta(self):
        self.dp.fill(0)

    @ti.kernel
    def calcDelta(self):
        mem = self.mem
        eps = 1e-9
        for ci in range(self.size):
            x,y,z     = self.Tris[ci]
            px,py,pz  = mem.newPos[x], mem.newPos[y], mem.newPos[z]
            invQ      = self.invQ[ci]        # constant material positon matrix, inversed

            p1 = py-px + self.dp[y] - self.dp[x]
            p2 = pz-px + self.dp[z] - self.dp[x]
            p  = mat2(p1, p2, byCol=True)      # world relative position matrix

            for i in ti.static(range(2)):
                for j in ti.static(range(i+1)):
                    # S = F^t*F;    G(Green - St Venant strain tensor) = S - I
                    fi = p @ getCol2(invQ, i)
                    fj = p @ getCol2(invQ, j)
                    Sij = fi.dot(fj)
                    # Derivatives of Sij
                    # d_p0_Sij = -SUM_k{d_pk_Sij}
                    d0, d1, d2 = vec3(), vec3(), vec3()
                    d1 = fj * invQ[0,i] + fi * invQ[0,j]
                    d2 = fj * invQ[1,i] + fi * invQ[1,j]
                    d0 = -(d1+d2)
                    # dp_k = -Lambda * invM_k * d_pk_Sij
                    # Lambda = 
                    #       (Sii - si^2) / SUM_k{invM_k * |d_pk_Sii|^2}    if i==j  ;    si: rest strech. typically 1
                    #                Sij / SUM_k{invM_k * |d_pk_Sii|^2}    if i<j
                    gradSum = d0.norm_sqr()*mem.invM[x] + d1.norm_sqr()*mem.invM[y] + \
                                d2.norm_sqr()*mem.invM[z]
                    vlambda = 0.
                    if abs(gradSum) > eps: 
                        if i == j:
                            vlambda = (Sij-1.) / gradSum * self.K[None]

                        else:
                            vlambda = Sij / gradSum * self.K[None]
                        self.dp[x] -= vlambda * d0 * mem.invM[x]
                        self.dp[y] -= vlambda * d1 * mem.invM[y]
                        self.dp[z] -= vlambda * d2 * mem.invM[z]
                    else:
                        print('WTF')

    @ti.kernel
    def applyDelta(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z  = self.Tris[i]
            mem.newPos[x] += self.dp[x] / self.w[x]
            mem.newPos[y] += self.dp[y] / self.w[y]
            mem.newPos[z] += self.dp[z] / self.w[z]
            #print('t', self.dp[x] / self.w[x], self.dp[y] / self.w[y], self.dp[z] / self.w[z])
    