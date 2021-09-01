'''
    https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf
'''
import taichi as ti # 0.7.29
import numpy as np
from scipy.spatial import Delaunay
from matrix import *
import time
ti.init(arch=ti.gpu)

# N = 20
# np.random.seed(42)
# x = .5 * np.random.rand(N) 
# y = .5 * np.random.rand(N) + 0.5
# z = .5 * np.random.rand(N)
# points = np.vstack([x, y, z]).T
points = [
    [.4,.45,.4],
    [.4,.45,.6],
    [.6,.45,.4],
    # [.6,.45,.6],
    # [.4,.55,.4],
    [.4,.55,.6],
    [.6,.55,.4],
    [.6,.55,.6]
]
a,b = 10, 10
deltaTheta = np.pi/a
deltaPhi = 2*np.pi/b
theta,phi = 0,0
for ring in range(a):
    theta += deltaTheta
    for p in range(b):
        phi += deltaPhi
        x = np.sin(theta) * np.cos(phi) / 5 + 0.5
        y = np.sin(theta) * np.sin(phi) / 5 + 0.5
        z = np.cos(theta) / 5 + 0.5
        points.append([x,y,z])
points = np.array(points)

N = len(points)
triangulation = Delaunay(points)
edges=[[], []]
for tet in triangulation.simplices:
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
K[None] = .5
TET           = ti.Vector.field(4, ti.i32, shape=NC) # Tetrahedron
InvRestMatrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=NC)
DELTA         = ti.Vector.field(3, ti.f32, shape=N) # postion correction cache
counts        = ti.field(ti.i32, shape=N)
# pbd
GRAVITY = ti.Vector([0., -9.8, 0.])
DeltaT  = 1 / 120

@ti.kernel
def initP():
    for i in ti.static(range(N)):
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
        TET[i] = triangulation.simplices[i]
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
        P[x] += DELTA[x] / counts[x]
        P[y] += DELTA[y] / counts[y]
        P[z] += DELTA[z] / counts[z]
        P[w] += DELTA[w] / counts[w]
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
    for _ in range(3):
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
    
gui = ti.GUI('Tetrahedron Constarint', background_color=0x112F41)
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
        print(K[None])
    if gui.is_pressed('q'):
        K[None] /= 1.1
        print(K[None])


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

    gui.lines(pos[edges[0]], pos[edges[1]], color=0xffeedd, radius=1*scale[None])
    gui.show()

    step()