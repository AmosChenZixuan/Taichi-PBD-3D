import taichi as ti # 0.7.29
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm

ti.init(arch=ti.gpu)

X    = ti.Vector.field(3, ti.f32, shape=12)
P    = ti.Vector.field(3, ti.f32, shape=12)
V    = ti.Vector.field(3, ti.f32, shape=12)
view = ti.Vector.field(2, ti.f32, shape=12)
edges = [
    [0,0,0,1,1,2,2,3,4,4,5,6, 1,1,2,3,4,2, 8,9,10,11, 8],
    [1,3,4,2,5,3,6,7,5,7,6,7, 3,4,5,4,6,7, 9,10,11,8, 10]
]
p = ti.field(ti.f32, shape=())
t = ti.field(ti.f32, shape=())
focus = ti.Vector.field(3, ti.f32, shape=())
scale = ti.field(ti.f32, shape=())

# shape
CM = ti.Vector.field(3, ti.f32, shape=())
Q  = ti.Vector.field(3, ti.f32, shape=8)
Q0 = ti.Vector.field(3, ti.f32, shape=8)
Apq = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
R   = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
ALPHA = ti.field(ti.f32, shape=())
ALPHA[None] = 1.

# pbd
GRAVITY = ti.Vector([0., -9.8, 0.])
DeltaT  = 1 / 120

@ti.kernel
def initP():
    for i in ti.static(range(8)):
        P[i] = X[i]

def initCamera():
    p[None] = 0.
    t[None] = 0.
    focus[None] = [0.5, 0.5, 0.5]
    scale[None] = 1.

def init():
    # cube
    X[0] = 0.4, 0.6, 0.4
    X[1] = 0.4, 0.6, 0.6
    X[2] = 0.6, 0.6, 0.6
    X[3] = 0.6, 0.6, 0.4
    X[4] = 0.4, 0.4, 0.4
    X[5] = 0.4, 0.4, 0.6
    X[6] = 0.6, 0.4, 0.6
    X[7] = 0.6, 0.4, 0.4
    initP()
    updateCM()
    updateQ()
    initQ0()
    #floor
    X[8]  = 0., 0., 0.
    X[9]  = 0., 0., 1.
    X[10] = 1., 0., 1.
    X[11] = 1., 0., 0.

#
# SHAPE MATCHING
#
@ti.kernel
def initQ0():
    for i in ti.static(range(8)):
        Q0[i] = Q[i]

@ti.kernel
def updateCM():
    cm = ti.Vector([0., 0., 0.])
    m = 0.
    for i in ti.static(range(8)):
        cm += P[i] * 1.
        m  += 1.
    cm /= m
    CM[None] = cm

@ti.kernel
def updateQ():
    for i in ti.static(range(8)):
        Q[i] = P[i] - CM[None]

@ti.kernel
def calcApq():
    A = ti.Vector([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ])
    for i in ti.static(range(8)):
        A += Q[i] @ Q0[i].transpose()
    Apq[None] = A

def calcR():
    A = Apq[None].value.to_numpy()
    S = sqrtm(A.T@A)
    R[None] = ti.Vector(A @ inv(S))

@ti.kernel
def updateDelta():
    for i in ti.static(range(8)):
        p = R[None] @ Q0[i] + CM[None]
        P[i] += (p - P[i]) * ALPHA[None] 

def solve():
    updateCM()
    updateQ()
    calcApq()
    calcR()
    updateDelta()

#
# PBD
#
@ti.kernel
def apply_force(mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
    for i in ti.static(range(8)):
        V[i] += GRAVITY * DeltaT
        P[i]  = X[i] + V[i] * DeltaT

@ti.kernel
def update():
    for i in ti.static(range(8)):
        V[i] = (P[i] - X[i]) / DeltaT * .99
        X[i] = P[i]

@ti.kernel
def box_confinement():
    for i in ti.static(range(8)):
        if P[i][1] < 0.:
            P[i][1] = 0.

def step():
    apply_force(0.,0.,0)
    box_confinement()
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

gui = ti.GUI('MPM3D', background_color=0x112F41)
initCamera()
init()
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
        ALPHA[None] *= 1.01
    if gui.is_pressed('q'):
        ALPHA[None] /= 1.01
    
    gui.circle(focus[None], radius=10*scale[None], color=0xff0000)
    gui.circles(pos, radius=5*scale[None], color=0x66ccff)
    gui.lines(pos[edges[0]], pos[edges[1]], color=0x66ccff, radius=1*scale[None])
    gui.show()

    step()