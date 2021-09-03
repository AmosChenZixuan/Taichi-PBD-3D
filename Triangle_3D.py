'''
    https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf
'''
import taichi as ti # 0.7.29
import numpy as np
from include.matrix import *
from include.data import vec3, mat2

import time

ti.init(arch=ti.gpu)

N = 3+4   # number of particles; plus one for floor

X    = ti.Vector.field(3, ti.f32, shape=N)
P    = ti.Vector.field(3, ti.f32, shape=N)
V    = ti.Vector.field(3, ti.f32, shape=N)
M    = ti.field(ti.f32, shape=N)              # INVERSED mass
view = ti.Vector.field(2, ti.f32, shape=N+1)  # plus one for focus
edges = [
    [0,0,1, 3,4,5,6, 3],
    [1,2,2, 4,5,6,3, 5]
]
p = ti.field(ti.f32, shape=())
t = ti.field(ti.f32, shape=())
focus = ti.Vector.field(3, ti.f32, shape=())
scale = ti.field(ti.f32, shape=())

# volume 
NC            = 1                            # number of constraint
K             = ti.field(ti.f32, shape=())   # stiffness
TRI           = ti.Vector.field(3, ti.i32, shape=NC) # Triangles
InvRestMatrix = ti.Matrix.field(2, 2, dtype=ti.f32, shape=NC)
DELTA         = ti.Vector.field(3, ti.f32, shape=(NC,3)) # postion correction cache
# pbd
GRAVITY = ti.Vector([0., -9.8, 0.])
DeltaT  = 1 / 120

@ti.kernel
def initP():
    for i in ti.static(range(N)):
        P[i] = X[i]

def initCamera():
    p[None] = 5.
    t[None] = 15.
    focus[None] = [0.5, 0.5, 0.5]
    scale[None] = 0.8

def init():
    V.fill(0)
    P.fill(0)
    K[None] = 1.
    # mesh
    X[0] = 0.5, 0.5, 0.5   ;M[0] = 1.
    X[1] = 0.45, 0.4, 0.45 ;M[1] = 1.
    X[2] = 0.55, 0.4, 0.45 ;M[2] = 0.
    initP()

    # constraint
    TRI[0] = 0, 1, 2
    initConstraint()
    
    #floor
    X[3] = 1., 0., 0.  ;M[3] = 0.
    X[4]  = 0., 0., 0. ;M[4] = 0.
    X[5]  = 0., 0., 1. ;M[5] = 0.
    X[6] = 1., 0., 1.  ;M[6] = 0.
    

#
# Volume Conservation
#
@ti.kernel
def initConstraint():
    for i in ti.static(range(NC)):
        x,y,z = TRI[i]
        col0 = tex(X[y]) - tex(X[x])
        col1 = tex(X[z]) - tex(X[x])

        InvRestMatrix[i]= mat2(col0, col1, byCol=True)

        InvRestMatrix[i] = InvRestMatrix[i].inverse()

@ti.func
def clearDelta(i):
    for j in ti.static(range(4)):
        DELTA[i,j] = vec3()

@ti.kernel
def calcDelta():
    eps = 1e-9
    for ci in range(NC):
        x,y,z     = TRI[ci]
        px,py,pz  = P[x], P[y], P[z]
        invQ           = InvRestMatrix[ci]        # constant material positon matrix, inversed
        clearDelta(ci)

        p1 = py-px #+ DELTA[ci,1] - DELTA[ci,0]
        p2 = pz-px #+ DELTA[ci,2] - DELTA[ci,0]
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
                gradSum = d0.norm_sqr()*M[x] + d1.norm_sqr()*M[y] + d2.norm_sqr()*M[z]
                vlambda = 0.
                if abs(gradSum) > eps: 
                    if i == j:
                        vlambda = (Sij-1.) / gradSum * K[None]

                    else:
                        vlambda = Sij / gradSum * K[None]
                    DELTA[ci,0] -= vlambda * d0 * M[x]
                    DELTA[ci,1] -= vlambda * d1 * M[y]
                    DELTA[ci,2] -= vlambda * d2 * M[z]
                    #print(vlambda * d0, vlambda * d1, vlambda * d2, vlambda * d3)

@ti.kernel
def applyDelta():
    for ci in range(NC):
        x,y,z  = TRI[ci]
        P[x] += DELTA[ci, 0] 
        P[y] += DELTA[ci, 1] 
        P[z] += DELTA[ci, 2] 
        #print(DELTA[ci, 0], DELTA[ci, 1], DELTA[ci, 2],DELTA[ci, 3])

def solve():
    '''
        Tetrahedral Constraints
    '''
    calcDelta()
    applyDelta()

    
#
# PBD
#
@ti.kernel
def apply_force(mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
    for i in range(N):
        if M[i] <= 0.: continue
        V[i] += GRAVITY * DeltaT
        P[i]  = X[i] + V[i] * DeltaT

        # mouse interaction
        if attract:
            P[0] = mouse_x, mouse_y, P[0][2]

@ti.kernel
def update():
    for i in range(N):
        if M[i] <= 0.: continue
        V[i] = (P[i] - X[i]) / DeltaT * .99
        X[i] = P[i]
    # pinned
    # X[7] = 0.6, 0.5, 0.4
    # P[7] = 0.6, 0.5, 0.4
    # V[7] = 0., 0., 0.

@ti.kernel
def box_confinement():
    for i in range(N):
        if P[i][1] < 0.:
            P[i][1] = 1e-4

def step():
    apply_force(mouse_pos[0], mouse_pos[1], attract)
    box_confinement()
    for _ in range(5):
        solve()
    update()

@ti.kernel
def project():
    phi   = p[None] * np.pi / 180.
    theta = t[None] * np.pi / 180.
    cp, sp = ti.cos(phi), ti.sin(phi)
    ct, st = ti.cos(theta), ti.sin(theta)

    for i in X:
        x,y,z = X[i] - focus[None]
        x, z = x * cp + z * sp, z * cp - x * sp
        u, v = x, y * ct + z * st
        view[i] = ti.Vector([u,v]) * scale[None] + ti.Vector([0.5, 0.5])
    # focus
    x,y,z = [0., 0., 0.]
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    view[N] = ti.Vector([u,v]) * scale[None] + ti.Vector([0.5, 0.5])

@ti.func
def tex(p3d):
    phi   = p[None] * np.pi / 180.
    theta = t[None] * np.pi / 180.
    cp, sp = ti.cos(phi), ti.sin(phi)
    ct, st = ti.cos(theta), ti.sin(theta)

    x,y,z = p3d - focus[None]
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return ti.Vector([u,v]) * scale[None] + ti.Vector([0.5, 0.5])
    
gui = ti.GUI('Triangle Constarint', background_color=0x112F41)
initCamera()
init()

attract = 0
mouse_pos = (0, 0)
while gui.running and not gui.get_event(gui.ESCAPE):
    project()
    pos = view.to_numpy()
    if gui.is_pressed(ti.GUI.LEFT):
        p[None] -= 1
    if gui.is_pressed(ti.GUI.RIGHT):
        p[None] += 1
    if gui.is_pressed(ti.GUI.UP):
        t[None] += 1
    if gui.is_pressed(ti.GUI.DOWN):
        t[None] -= 1
    if gui.is_pressed(']'):
        scale[None] *= 1.01
    if gui.is_pressed('['):
        scale[None] /= 1.01
    if gui.is_pressed('w'):
        focus[None][1] += 0.01
    if gui.is_pressed('a'):
        focus[None][0] -= 0.01
    if gui.is_pressed('s'):
        focus[None][1] -= 0.01
    if gui.is_pressed('d'):
        focus[None][0] += 0.01
    if gui.is_pressed('r'):
        init()
    if gui.is_pressed('c'):
        initCamera()
    if gui.is_pressed('e'):
        if K[None] <= 1.:
            K[None] *= 1.1
    if gui.is_pressed('q'):
        K[None] /= 1.1


    if gui.is_pressed(ti.GUI.RMB):
        mouse_pos = gui.get_cursor_pos()
        attract = 1
    elif gui.is_pressed(ti.GUI.LMB):
        mouse_pos = gui.get_cursor_pos()
        attract = -1
    else:
        attract = 0

    
    gui.circles(pos[:N], radius=5*scale[None], color=0x66ccff)
    gui.circle(pos[N], radius=1*scale[None], color=0xff0000)

    gui.lines(pos[edges[0]], pos[edges[1]], color=0x66ccff, radius=1*scale[None])
    gui.show()

    step()