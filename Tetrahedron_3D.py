import taichi as ti # 0.7.29
import numpy as np
from matrix3 import *
from matrix import *

import time

ti.init(arch=ti.gpu)

N = 4+4   # number of particles; plus one for floor

X    = ti.Vector.field(3, ti.f32, shape=N)
P    = ti.Vector.field(3, ti.f32, shape=N)
V    = ti.Vector.field(3, ti.f32, shape=N)
M    = ti.field(ti.f32, shape=N)
view = ti.Vector.field(2, ti.f32, shape=N+1)  # plus one for focus
edges = [
    [0,0,0, 1,2,3, 4,5,6,7, 4],
    [1,2,3, 2,3,1, 5,6,7,4, 6]
]
p = ti.field(ti.f32, shape=())
t = ti.field(ti.f32, shape=())
focus = ti.Vector.field(3, ti.f32, shape=())
scale = ti.field(ti.f32, shape=())

# volume 
NC            = 1                            # number of constraint
K             = ti.field(ti.f32, shape=())   # stiffness
TET           = ti.Vector.field(4, ti.i32, shape=NC) # Tetrahedron
InvRestMatrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=NC)
DELTA         = ti.Vector.field(3, ti.f32, shape=(NC,4)) # postion correction cache
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
    K[None] = .5
    # mesh
    X[0] = 0.5, 0.5, 0.5 ;M[0] = 1.
    X[1] = 0.45, 0.4, 0.45 ;M[1] = 1.
    X[2] = 0.55, 0.4, 0.45 ;M[2] = 1.
    X[3] = 0.5, 0.4, 0.55 ;M[3] = 1.
    initP()

    # constraint
    TET[0] = 0, 1, 2, 3
    initConstraint()
    
    #floor
    X[4]  = 0., 0., 0. ;M[4] = 0.
    X[5]  = 0., 0., 1. ;M[5] = 0.
    X[6] = 1., 0., 1.  ;M[6] = 0.
    X[7] = 1., 0., 0.  ;M[7] = 0.

#
# Volume Conservation
#
@ti.kernel
def initConstraint():
    for i in ti.static(range(NC)):
        x,y,z,w = TET[i]
        col0 = X[y] - X[x]
        col1 = X[z] - X[x]
        col2 = X[w] - X[x]

        InvRestMatrix[i]= mat3(col0, col1, col2).transpose()

        InvRestMatrix[i] = InvRestMatrix[i].inverse()
 
@ti.func
def clearDelta(i):
    for j in ti.static(range(4)):
        DELTA[i,j] = vec3()

@ti.kernel
def calcDelta():
    eps = 1e-9
    for ci in range(NC):
        x,y,z,w     = TET[ci]
        px,py,pz,pw = P[x], P[y], P[z], P[w]  
        invMat      = InvRestMatrix[ci]
        clearDelta(ci)
        c0 = getCol0(invMat)
        c1 = getCol1(invMat)
        c2 = getCol2(invMat)

        for i in ti.static(range(3)):
            for j in ti.static(range(i+1)):
                p = mat3()
                p = setCol0(p, py-px) #+ DELTA[ci,1] - DELTA[ci,0])
                p = setCol1(p, pz-px) #+ DELTA[ci,2] - DELTA[ci,0])
                p = setCol2(p, pw-px) #+ DELTA[ci,3] - DELTA[ci,0])

                # fi = p*c[i] ; fj = p*c[j]
                fi, fj = vec3(), vec3()
                if i == 0: fi = p @ c0
                elif i==1: fi = p @ c1
                else     : fi = p @ c2
                if j == 0: fj = p @ c0
                elif j==1: fj = p @ c1
                else     : fj = p @ c2
                #print(i,j, fi, fj)
                Sij = fi.dot(fj)
                #print(Sij)

                d0, d1, d2, d3 = vec3(), vec3(), vec3(), vec3()
                # for k in range(3)
                d1 = fj * InvRestMatrix[ci][0,i] + fi * InvRestMatrix[ci][0,j]; d0 -= d1
                d2 = fj * InvRestMatrix[ci][1,i] + fi * InvRestMatrix[ci][1,j]; d0 -= d2
                d3 = fj * InvRestMatrix[ci][2,i] + fi * InvRestMatrix[ci][2,j]; d0 -= d3
                #print(i, j, d0, d1, d2, d3)

                vlambda = d0.norm_sqr() + d1.norm_sqr() + d2.norm_sqr() + d3.norm_sqr()
                if abs(vlambda) > eps: 
                    if i == j:
                        vlambda = (Sij-1.) / vlambda * K[None]

                    else:
                        vlambda = Sij / vlambda * K[None]

                    DELTA[ci,0] -= vlambda * d0
                    DELTA[ci,1] -= vlambda * d1
                    DELTA[ci,2] -= vlambda * d2
                    DELTA[ci,3] -= vlambda * d3
                    #print(vlambda * d0, vlambda * d1, vlambda * d2, vlambda * d3)
                else:
                    print("wtf")

@ti.kernel
def applyDelta():
    for ci in range(NC):
        x,y,z,w  = TET[ci]
        P[x] += DELTA[ci, 0] 
        P[y] += DELTA[ci, 1] 
        P[z] += DELTA[ci, 2] 
        P[w] += DELTA[ci, 3]
        #print(DELTA[ci, 0], DELTA[ci, 1], DELTA[ci, 2],DELTA[ci, 3])

def solve():
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
            P[i][1] = 0.

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
    
gui = ti.GUI('Tetrahedron Volume Conservation', background_color=0x112F41)
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