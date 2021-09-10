import taichi as ti
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from include import *

def angle_axis(direction, angle):
        result = np.array([0., 0., 0., 1.])
        #angle = np.radians(angle)
        sin = np.sin(angle/2)
        cos = np.cos(angle/2)

        result[:-1] = direction * sin
        result[3]   = cos
        return result

def quaternion(float4):
    try:
        return Rotation.from_quat(float4)
    except:
        return Rotation.from_quat(np.array([0., 0., 0., 1.]))

def qmul(q1, q2):
    # quaternion multiplication
    x,y,z,w = 0,1,2,3
    result = np.zeros(4)
    result[0] = q1[w]*q2[x] + q1[x]*q2[w] + q1[y]*q2[z] - q1[z]*q2[y]  # i
    result[1] = q1[w]*q2[y] - q1[x]*q2[z] + q1[y]*q2[w] + q1[z]*q2[x]  # j
    result[2] = q1[w]*q2[z] + q1[x]*q2[y] - q1[y]*q2[x] + q1[z]*q2[w]  # k
    result[3] = q1[w]*q2[w] - q1[x]*q2[x] - q1[y]*q2[y] - q1[z]*q2[z]  # l
    return result

@ti.data_oriented
class ShapeMatchingSolver:
    def __init__(self, memory: Memory, nParticles):
        self.mem  = memory
        self.size = nParticles
        self.ptr  = field(nParticles, 1, ti.i32)

        self.CM   = field((), 3, ti.f32)            # center of mass
        self.Q    = field(nParticles, 3, ti.f32)    # position matrix
        self.Q0   = field(nParticles, 3, ti.f32)    # rest postion mat
        self.Apq  = field((), (3,3), ti.f32)        # deformation matrix
        self.R    = field((), (3,3), ti.f32)        # rotation matrix
        self.quat = np.array([0., 0., 0., 1.])      # rotation as quaternion 
        self.ALPHA = field((), 1, ti.f32)           # stiffness

    def reset(self):
        self.ALPHA[None] = .1
        self.quat = np.array([0., 0., 0., 1.])

    def update(self, i, idx):
        self.ptr[i] = idx

    def init(self):
        self.updateCM()
        self.updateQ()
        self.initQ0()

    #@timeThis
    def solve(self):
        self.updateCM()
        self.updateQ()
        self.calcApq()
        self.calcR_extract()
        self.updateDelta() 

    ################### Private Methods #####################

    @ti.kernel
    def initQ0(self):
        for x in range(self.size):
            i = self.ptr[x]
            self.Q0[i] = self.Q[i]

    @ti.kernel
    def updateCM(self):
        mem = self.mem
        cm  = vec3()
        m   = 0.
        for x in range(self.size):
            i = self.ptr[x]
            mass = 1. if mem.invM[i]==0. else 1./mem.invM[i]
            cm += mem.newPos[i] * mass
            m  += mass
        self.CM[None] = cm / m

    @ti.kernel
    def updateQ(self):
        mem = self.mem
        for x in range(self.size):
            i = self.ptr[x]
            self.Q[i] = mem.newPos[i] - self.CM[None]
    
    @ti.kernel
    def calcApq(self):
        A = mat3()
        for x in range(self.size):
            i = self.ptr[x]
            A += self.Q[i] @ self.Q0[i].transpose()
        self.Apq[None] = A

    def calcR_extract(self):
        A = self.Apq[None].value.to_numpy()
        q = self.quat
        for _ in range(3):
            rot = quaternion(q)
            r = rot.as_matrix()
            omega_numerator = np.cross(r[:,0], A[:,0]) + \
                            np.cross(r[:,1], A[:,1]) + \
                            np.cross(r[:,2], A[:,2])
            omega_denom = 1. / (abs(
                np.dot(r[:,0], A[:,0]) + \
                np.dot(r[:,1], A[:,1]) + \
                np.dot(r[:,2], A[:,2])
            ) + 1.e-9)
            omega = omega_numerator * omega_denom
            w = norm(omega)
            if w < 1.e-9:
                break
            q = qmul(angle_axis(omega/w, w), q)  # very important
            q = q / norm(q)
        self.quat = q
        self.R[None] = ti.Vector(quaternion(q).as_matrix())

    @ti.kernel
    def updateDelta(self):
        mem = self.mem
        for x in range(self.size):
            i = self.ptr[x]
            p = self.R[None] @ self.Q0[i] + self.CM[None]
            mem.newPos[i] += (p - mem.newPos[i]) * self.ALPHA[None] 

    