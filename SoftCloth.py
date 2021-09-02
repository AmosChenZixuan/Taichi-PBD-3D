'''
    https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf
'''
import taichi as ti # 0.7.29
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
from matrix import *
from cloth import createRandomCloth, createRectCloth
import time
ti.init(arch=ti.gpu)

@ti.pyfunc
def tex(p3d):
    cp, sp = ti.cos(0), ti.sin(0)
    ct, st = ti.cos(0), ti.sin(0)

    x,y,z = p3d - ti.Vector([.5,.5,.5])
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return ti.Vector([u,v]) * 1. + ti.Vector([0.5, 0.5])

# build scene objects
leftbtm = (.25, .25, .5)
points = []
np.random.seed(0)
points.extend(createRectCloth(25,25,0.02, 0.02, leftbtm))


points2d = np.array([tex(x) for x in points])
N = len(points)
triangulation = Delaunay(points2d)
edges=[[], []]
for tet in triangulation.simplices:
    for i in range(3):
        for j in range(i+1, 3):
            edges[0].append(tet[i])
            edges[1].append(tet[j])

# Allocate memory
N = N+4   # number of particles; plus one for floor
X    = ti.Vector.field(3, ti.f32, shape=N)
P    = ti.Vector.field(3, ti.f32, shape=N)
V    = ti.Vector.field(3, ti.f32, shape=N)
M    = ti.field(ti.f32, shape=N)              # INVERSED mass
view = ti.Vector.field(2, ti.f32, shape=N+1)  # plus one for focus
# add floor
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])
# camera
p = ti.field(ti.f32, shape=())
t = ti.field(ti.f32, shape=())
focus = ti.Vector.field(3, ti.f32, shape=())
scale = ti.field(ti.f32, shape=())
# volume 
NC            = len(triangulation.simplices)           # number of constraint
K             = ti.field(ti.f32, shape=())   # stiffness
K[None] = 1.
TRI           = ti.Vector.field(3, ti.i32, shape=NC) # Triangles
InvRestMatrix = ti.Matrix.field(2, 2, dtype=ti.f32, shape=NC)
DELTA         = ti.Vector.field(3, ti.f32, shape=N) # postion correction cache
counts        = ti.field(ti.i32, shape=N)
# pbd
GRAVITY = ti.Vector([0., -9.8, 0.])
DeltaT  = 1 / 240
iter    = ti.field(ti.i32, shape=())   # solver iterations
iter[None] = 3
# sim
paused  = False
picked  = -1
mouse_pos = (0, 0)
RMB     = False

@ti.kernel
def initP():
    for i in range(N):
        P[i] = X[i]

def initCamera():
    p[None] = 5.
    t[None] = 1.
    focus[None] = [0.5, 0.5, 0.5]
    scale[None] = 0.8

def init():
    V.fill(0)
    P.fill(0)
    counts.fill(0)
    # mesh
    for i in range(N-4):
        X[i] = points[i]
        M[i] = 1.
    #M[1] = 0.
    initP()

    # constraint
    for i in range(NC):
        TRI[i] = triangulation.simplices[i]
        x,y,z = TRI[i]
        counts[x] += 3
        counts[y] += 3
        counts[z] += 3
    initConstraint()
    
    #floor
    X[N-4]  = 0., 0., 0. ;M[N-4] = 0.
    X[N-3]  = 0., 0., 1. ;M[N-3] = 0.
    X[N-2] = 1., 0., 1.  ;M[N-2] = 0.
    X[N-1] = 1., 0., 0.  ;M[N-1] = 0.

#
# Volume Conservation
#
@ti.kernel
def initConstraint():
    for i in range(NC):
        x,y,z = TRI[i]
        col0 = tex(X[y]) - tex(X[x])
        col1 = tex(X[z]) - tex(X[x])

        InvRestMatrix[i]= mat2(col0, col1, byCol=True)

        InvRestMatrix[i] = InvRestMatrix[i].inverse()

def clearDelta():
    DELTA.fill(0)

@ti.kernel
def calcDelta():
    eps = 1e-9
    for ci in range(NC):
        x,y,z     = TRI[ci]
        px,py,pz  = P[x], P[y], P[z]
        invQ           = InvRestMatrix[ci]        # constant material positon matrix, inversed

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
                    DELTA[x] -= vlambda * d0 * M[x]
                    DELTA[y] -= vlambda * d1 * M[y]
                    DELTA[z] -= vlambda * d2 * M[z]
                    #print(vlambda * d0, vlambda * d1, vlambda * d2, vlambda * d3)
@ti.kernel
def applyDelta():
    for ci in range(NC):
        x,y,z  = TRI[ci]
        P[x] += min(DELTA[x] / counts[x], .01)
        P[y] += min(DELTA[y] / counts[y], .01)
        P[z] += min(DELTA[z] / counts[z], .01)
        #print(DELTA[ci, 0], DELTA[ci, 1], DELTA[ci, 2],DELTA[ci, 3])

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

def pick(mouse_pos):
    pos = view.to_numpy() - mouse_pos
    pos *= -1
    dists = np.array([norm(v) for v in pos])
    closest = int(np.argmin(dists))
    return closest if dists[closest] < 0.1 else -1

@ti.kernel
def apply_force(mouse_x: ti.f32, mouse_y: ti.f32, idx: ti.i32):
    for i in range(N):
        if M[i] <= 0.: continue
        V[i] += GRAVITY * DeltaT
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

def step():
    if not paused:
        apply_force(mouse_pos[0], mouse_pos[1], picked)
        box_confinement()
        for _ in range(iter[None]):
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
    
###
###
###
gui = ti.GUI('Triangle Constarint', background_color=0x112F41)
initCamera()
init()
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS, ti.GUI.MOTION):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused = not paused
        elif e.key == 'r':
            init()
        elif e.key == 'c':
            initCamera()
        elif e.key == 't':
            paused = False
            step()
            paused = True
        # stiffness
        elif e.key == 'e':
            K[None] *= 1.05
        elif e.key == 'q':
            K[None] /= 1.05
        # scale/zoom
        if e.key == ti.GUI.WHEEL:
            scale[None] *= 1.01 if e.delta.y>0 else 1/1.01
        # camera angle
        if e.key == ti.GUI.RMB:
            RMB = not RMB
            mouse_pos = e.pos
        if e.type==ti.GUI.MOTION and RMB:
            xdelta, ydelta = (ti.Vector(e.pos)-ti.Vector(mouse_pos))
            p[None] -= xdelta * 100
            t[None] -= ydelta * 100
            mouse_pos = e.pos
        # pin particle
        if e.key == 'p':
            idx = picked if picked >=0 else pick(e.pos)
            if idx >= 0:
                M[idx] = 0. if M[idx] != 0. else 1.
    # no delay control
    # camera angle
    if gui.is_pressed(ti.GUI.LEFT):
        p[None] -= 1
    elif gui.is_pressed(ti.GUI.RIGHT):
        p[None] += 1
    elif gui.is_pressed(ti.GUI.UP):
        t[None] += 1
    elif gui.is_pressed(ti.GUI.DOWN):
        t[None] -= 1
    if gui.is_pressed('w'):
        focus[None][1] += 0.01
    elif gui.is_pressed('a'):
        focus[None][0] -= 0.01
    elif gui.is_pressed('s'):
        focus[None][1] -= 0.01
    elif gui.is_pressed('d'):
        focus[None][0] += 0.01
    # mouse interaction
    if gui.is_pressed(ti.GUI.LMB):
        mouse_pos = gui.get_cursor_pos()
        if picked == -1:
            picked = pick(mouse_pos)
    else:
        picked = -1
    # iteration
    if gui.is_pressed(']'):
        iter[None] += 1
    elif gui.is_pressed('['):
        iter[None] -= 1


    # render
    project()
    pos = view.to_numpy()
    gui.circles(pos[:N], radius=1*scale[None], color=0x66ccff)
    gui.circle(pos[N], radius=1*scale[None], color=0xff0000)
    gui.lines(pos[edges[0]], pos[edges[1]], color=0xffeedd, radius=1*scale[None])
    gui.text(content=f'Stiffness={K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={iter[None]}',pos=(0,0.9), color=0xffffff)
    gui.show()
    # sim
    step()