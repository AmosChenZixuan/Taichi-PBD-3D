'''
    https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf
'''
import taichi as ti # 0.7.29
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

from include import *
from UI import Camera, EventHandler
ti.init(arch=ti.gpu)

# build scene objects
import pygalmesh
class Heart(pygalmesh.DomainBase):
    def __init__(self):
        super().__init__()

    def eval(self, x):
        return (
            (x[0] ** 2 + 9.0 / 4.0 * x[2] ** 2 + x[1] ** 2 - 1) ** 3
            - x[0] ** 2 * x[1] ** 3
            - 9.0 / 80.0 * x[2] ** 2 * x[1] ** 3
        )

    def get_bounding_sphere_squared_radius(self):
        return 10.0

d = Heart()
mesh = pygalmesh.generate_mesh(d, max_cell_circumradius=0.3, verbose=False)

points = np.array(mesh.points)/10 + 0.5
N = len(points)
triangulation = mesh.cells[1].data
edges=[[], []]
for tet in triangulation:
    for i in range(4):
        for j in range(i+1, 4):
            edges[0].append(tet[i])
            edges[1].append(tet[j])

# Allocate memory
N = N+4   # number of particles; plus one for floor
X    = ti.Vector.field(3, ti.f32, shape=N)
P    = ti.Vector.field(3, ti.f32, shape=N)
V    = ti.Vector.field(3, ti.f32, shape=N)
M    = ti.field(ti.f32, shape=N)              # INVERSED mass
#view = ti.Vector.field(2, ti.f32, shape=N+1)  # plus one for focus
# add floor
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])
# camera
camera = Camera(focus=(.5, .5,.5), angle=(5., -1.), scale=.8)
# volume 
NC            = len(triangulation)           # number of constraint
K             = ti.field(ti.f32, shape=())   # stiffness
K[None] = .8
TET           = ti.Vector.field(4, ti.i32, shape=NC) # Tetrahedron
InvRestMatrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=NC)
DELTA         = ti.Vector.field(3, ti.f32, shape=N) # postion correction cache
counts        = ti.field(ti.i32, shape=N)
# pbd
GRAVITY = ti.Vector.field(3, ti.f32, shape=())
GRAVITY[None] = ti.Vector([0., -9.8, 0.])
DeltaT  = 1 / 240
iter    = ti.field(ti.i32, shape=())   # solver iterations
iter[None] = 3

# shape
CM = ti.Vector.field(3, ti.f32, shape=())
Q  = ti.Vector.field(3, ti.f32, shape=N)
Q0 = ti.Vector.field(3, ti.f32, shape=N)
Apq = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
R   = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
ALPHA = ti.field(ti.f32, shape=())
ALPHA[None] = .5

@ti.kernel
def initP():
    for i in range(N):
        P[i] = X[i]

def init():
    V.fill(0)
    P.fill(0)
    counts.fill(0)
    Apq.fill(0)
    R.fill(0)
    # mesh
    for i in range(N-4):
        X[i] = points[i]
        M[i] = 1.
    #M[1] = 0.
    initP()
    updateCM()
    updateQ()
    initQ0()

    # constraint
    for i in range(NC):
        TET[i] = triangulation[i]
        x,y,z,w = TET[i]
        counts[x] += 4
        counts[y] += 4
        counts[z] += 4
        counts[w] += 4
    initConstraint()
    
    #floor
    X[N-4]  = 0., 0., 0. ;M[N-4] = 0.
    X[N-3]  = 0., 0., 1. ;M[N-3] = 0.
    X[N-2] = 1., 0., 1.  ;M[N-2] = 0.
    X[N-1] = 1., 0., 0.  ;M[N-1] = 0.

#
# SHAPE MATCHING
#
@ti.kernel
def initQ0():
    for i in range(N-4):
        Q0[i] = Q[i]

@ti.kernel
def updateCM():
    cm = ti.Vector([0., 0., 0.])
    m = 0.
    for i in range(N-4):
        cm += P[i] * 1.
        m  += 1.
    cm /= m
    CM[None] = cm

@ti.kernel
def updateQ():
    for i in range(N-4):
        Q[i] = P[i] - CM[None]

@ti.kernel
def calcApq():
    A = ti.Vector([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    for i in range(N-4):
        A += Q[i] @ Q0[i].transpose()
    Apq[None] = A

def angle_axis(direction, angle):
    result = np.array([0., 0., 0., 1.])
    #angle = np.radians(angle)
    sin = np.sin(angle/2)
    cos = np.cos(angle/2)

    result[:-1] = direction * sin
    result[3]   = cos
    return result

def quaternion(float4):
    return Rotation.from_quat(float4)

def qmul(q1, q2):
    # quaternion multiplication
    x,y,z,w = 0,1,2,3
    result = np.zeros(4)
    result[0] = q1[w]*q2[x] + q1[x]*q2[w] + q1[y]*q2[z] - q1[z]*q2[y]  # i
    result[1] = q1[w]*q2[y] - q1[x]*q2[z] + q1[y]*q2[w] + q1[z]*q2[x]  # j
    result[2] = q1[w]*q2[z] + q1[x]*q2[y] - q1[y]*q2[x] + q1[z]*q2[w]  # k
    result[3] = q1[w]*q2[w] - q1[x]*q2[x] - q1[y]*q2[y] - q1[z]*q2[z]  # l
    return result


def calcR_extract():
    A = Apq[None].value.to_numpy()
    R_prev = R[None].value.to_numpy()
    if norm(R_prev) == 0.:
        q = np.array([0., 0., 0., 1.])
    else:
        q = Rotation.from_matrix(R_prev).as_quat()

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
    R[None] = ti.Vector(quaternion(q).as_matrix())


@ti.kernel
def updateDelta():
    for i in range(N-4):
        p = R[None] @ Q0[i] + CM[None]
        P[i] += (p - P[i]) * ALPHA[None] 

#@timeThis
def solveShape():
    updateCM()
    updateQ()
    calcApq()
    calcR_extract()
    updateDelta()

#
# Volume Conservation
#
@ti.kernel
def initConstraint():
    for i in range(NC):
        x,y,z,w = TET[i]
        col0 = X[y] - X[x]
        col1 = X[z] - X[x]
        col2 = X[w] - X[x]

        InvRestMatrix[i]= mat3(col0, col1, col2, byCol=True)

        InvRestMatrix[i] = InvRestMatrix[i].inverse()
        #print(InvRestMatrix[i])

def clearDelta():
    DELTA.fill(0)

@ti.kernel
def calcDelta():
    eps = 1e-9
    for ci in range(NC):
        x,y,z,w     = TET[ci]
        px,py,pz,pw = P[x], P[y], P[z], P[w]  
        invQ        = InvRestMatrix[ci]        # constant material positon matrix, inversed

        p1 = py-px + DELTA[y] - DELTA[x]
        p2 = pz-px + DELTA[z] - DELTA[x]
        p3 = pw-px + DELTA[w] - DELTA[x]
        p  = mat3(p1, p2, p3, byCol=True)      # world relative position matrix

        for i in ti.static(range(3)):
            for j in ti.static(range(i+1)):
                # S = F^t*F;    G(Green - St Venant strain tensor) = S - I
                fi = p @ getCol(invQ, i)
                fj = p @ getCol(invQ, j)
                Sij = fi.dot(fj)
                # Derivatives of Sij
                # d_p0_Sij = -SUM_k{d_pk_Sij}
                d0, d1, d2, d3 = vec3(), vec3(), vec3(), vec3()
                d1 = fj * invQ[0,i] + fi * invQ[0,j]
                d2 = fj * invQ[1,i] + fi * invQ[1,j]
                d3 = fj * invQ[2,i] + fi * invQ[2,j]
                d0 = -(d1+d2+d3)
                # dp_k = -Lambda * invM_k * d_pk_Sij
                # Lambda = 
                #       (Sii - si^2) / SUM_k{invM_k * |d_pk_Sii|^2}    if i==j  ;    si: rest strech. typically 1
                #                Sij / SUM_k{invM_k * |d_pk_Sii|^2}    if i<j
                gradSum = d0.norm_sqr()*M[x] + d1.norm_sqr()*M[y] + d2.norm_sqr()*M[z] + d3.norm_sqr()*M[w]
                vlambda = 0.
                if abs(gradSum) > eps: 
                    if i == j:
                        vlambda = (Sij-1.) / gradSum * K[None]

                    else:
                        vlambda = Sij / gradSum * K[None]
                    DELTA[x]  -= vlambda * d0 * M[x]
                    DELTA[y]  -= vlambda * d1 * M[y]
                    DELTA[z]  -= vlambda * d2 * M[z]
                    DELTA[w]  -= vlambda * d3 * M[w]
                    #print(vlambda * d0, vlambda * d1, vlambda * d2, vlambda * d3)
                else:
                    print('WTF')

@ti.kernel
def applyDelta():
    for ci in range(NC):
        x,y,z,w  = TET[ci]
        P[x] += min(DELTA[x] / counts[x], .01) / 4
        P[y] += min(DELTA[y] / counts[y], .01) / 4
        P[z] += min(DELTA[z] / counts[z], .01) / 4
        P[w] += min(DELTA[w] / counts[w], .01) / 4
        #print(DELTA[ci, 0], DELTA[ci, 1], DELTA[ci, 2],DELTA[ci, 3])

#@timeThis
def solve():
    '''
        Tetrahedral Constraints
    '''
    clearDelta()
    calcDelta()
    applyDelta()
    
#
# PBD
#
def pick(pos, mouse_pos):
    pos = (pos - mouse_pos) * -1              # mouse-pos = -(pos-mouse)
    dists = np.array([norm(v) for v in pos])
    closest = int(np.argmin(dists))
    return closest if dists[closest] < 0.1 else -1

@ti.kernel
def apply_force(mouse_x: ti.f32, mouse_y: ti.f32, idx: ti.i32):
    for i in range(N):
        if M[i] <= 0.: continue
        V[i] += GRAVITY[None] * DeltaT
        P[i]  = X[i] + V[i] * DeltaT

        # mouse interaction
        if idx>=0:
            P[idx] = mouse_x, mouse_y, P[idx][2]

@ti.kernel
def update():
    for i in range(N):
        if M[i] <= 0.: continue
        V[i] = (P[i] - X[i]) / DeltaT * .99
        X[i] = P[i]

@ti.kernel
def box_confinement():
    for i in range(N):
        if P[i][1] < 0.:
            P[i][1] = 1e-4

def step(paused, mouse_pos, picked):
    if not paused:
        apply_force(mouse_pos[0], mouse_pos[1], picked)
        box_confinement()
        for _ in range(iter[None]):
            solve()
        solveShape()
        update()

@ti.kernel
def project(p3: ti.ext_arr(), p2: ti.ext_arr()):
    for i in range(p3.shape[0]):
        x,y,z = p3[i,0], p3[i,1], p3[i,2]
        u,v   = camera.project(vec3(x,y,z))
        p2[i,0] = u
        p2[i,1] = v
    
###
###
###
gui = ti.GUI('Tetrahedron Constarint', res=(1000,1000), background_color=0x112F41)
eventHandler = EventHandler(gui)
init()
while gui.running:
    pos3 = X.to_numpy()
    pos2 = np.zeros((len(pos3), 2))
    project(pos3, pos2)

    eventHandler.regularReact(camera=camera, stiffness_field=K, iteration_field=iter, mass_field=M,
                            pos2d=pos2,
                            init_method=init, step_method=step, pick_method=pick
                            )
    eventHandler.fastReact(camera=camera, 
                        pos2d=pos2,
                        pick_method=pick)
    if gui.is_pressed(","):
        ALPHA[None] /= 1.1
    elif gui.is_pressed('.'):
        ALPHA[None] *= 1.1
    if gui.is_pressed("="):
        GRAVITY[None][1] /= 1.1
    elif gui.is_pressed('-'):
        GRAVITY[None][1] *= 1.1
    
    # render
    scale = camera.getScale()
    gui.circles(pos2, radius=2*scale, color=0xff0000)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xff0000, radius=.5*scale)
    gui.text(content=f'Stiffness={K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={iter[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Shape={ALPHA[None]}',pos=(0,0.85), color=0xffffff)
    gui.text(content=f'Gravity={GRAVITY[None].value}',pos=(0,0.8), color=0xffffff)
    gui.show()
    # sim
    step(eventHandler.paused, eventHandler.mouse_pos, eventHandler.picked)