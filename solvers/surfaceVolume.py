'''
    https://animation.rwth-aachen.de/media/papers/2011-SCA-RobustRealTimeDeformation.pdf
'''
import taichi as ti
from include import *


@ti.data_oriented
class SurfaceBasedVolumeSolver:
    def __init__(self, memory: Memory, nParticles, nTris, retStf=1.):
        self.mem  = memory
        self.size = nTris
        self.retStf = retStf

        self.K    = field((), 1, ti.f32)        # stiffness
        self.V    = field((), 1, ti.f32)        # curr Volume
        self.V3   = field((), 1, ti.f32)        # rest Volume const; 3 * V0
        self.Tris = field(nTris, 3, ti.i32)     # vertices of triangles
        # self.norm = field(nTris, 3, ti.f32)     # triangles norms
        # self.A    = field(nTris, 1, ti.f32)     # triangles areas
        self.w    = field(nParticles, 1, ti.f32)  # weights
        self.nSum = field(nParticles, 3, ti.f32)  # sum of the area weighted normals of all triangles containing the i-th particle.
        

    def reset(self):
        self.K[None]  = self.retStf
        self.V[None] = 0.
        self.V3[None] = 0.

    def update(self, i, x, y, z):
        self.Tris[i] = x,y,z

    def init(self):
        self.updateTris()
        self.updateVol()
        self.V3[None] = 3 * self.V[None]
        print(self.V3[None])
        t=0

    
    #@timeThis
    def solve(self):
        self.clearData()
        self.updateTris()
        self.updateVol()
        self.project()

    ################### Private Methods #####################
    def clearData(self):
        self.nSum.fill(0.)
    

    @ti.kernel
    def updateTris(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z     = self.Tris[i]
            px,py,pz  = mem.newPos[x], mem.newPos[y], mem.newPos[z]
            n         = (py-px).cross(pz-px)
            
            area      = self.calcArea(px,py,pz)
            AN        = area * n 
            if AN.norm() <= 0. or AN.norm() > .01:
                print(px,py,pz, n, area)
                continue
            # self.norm[i] = n
            # self.A[i]    = area
            self.nSum[x]+= AN
            self.nSum[y]+= AN
            self.nSum[z]+= AN


    @ti.kernel
    def updateVol(self):
        '''
            1/3. * sum_i{ xi^T * weighted_norm_i}
        '''
        self.V[None] = 0.
        mem = self.mem
        for i in self.nSum:
            self.V[None] += (mem.newPos[i].transpose() @ self.nSum[i])[0,0]
        self.V[None] /= 3.
        #print(self.V[None])

    @ti.kernel
    def project(self):
        mem = self.mem
        for i in range(self.size):
            x,y,z     = self.Tris[i]
            C  = self.V[None] - self.V3[None]
            d0 = self.nSum[x] / 3.
            d1 = self.nSum[y] / 3.
            d2 = self.nSum[z] / 3.

            gradSum = d0.norm_sqr() + d1.norm_sqr() + d2.norm_sqr()
            #print(gradSum, self.V[None], self.V3[None])
            if 1 > abs(gradSum) > 1e-9:
                mem.newPos[x] -= C * d0 / gradSum * self.K[None]
                mem.newPos[y] -= C * d1 / gradSum * self.K[None]
                mem.newPos[z] -= C * d2 / gradSum * self.K[None]

    @ti.func
    def calcArea(self, x,y,z):
        a,b,c = (x-y).norm(), (y-z).norm(), (z-x).norm()
        s     = (a+b+c)/2.
        return (s*(s-a)*(s-b)*(s-c)) ** 0.5  
