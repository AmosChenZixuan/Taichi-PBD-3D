import taichi as ti # 0.7.29
import numpy as np

import time

ti.init(arch=ti.gpu)

N = 4+4

X    = ti.Vector.field(3, ti.f32, shape=N)
P    = ti.Vector.field(3, ti.f32, shape=N)
V    = ti.Vector.field(3, ti.f32, shape=N)
M    = ti.field(ti.f32, shape=N)
view = ti.Vector.field(2, ti.f32, shape=N+1)
edges = [
    [0,0,0, 1,2,3, 4,5,6,7, 4],
    [1,2,3, 2,3,1, 5,6,7,4, 6]
]
p = ti.field(ti.f32, shape=())
t = ti.field(ti.f32, shape=())
focus = ti.Vector.field(3, ti.f32, shape=())
scale = ti.field(ti.f32, shape=())

# volume 
K = ti.field(ti.f32, shape=())   # k pressure
TRI = ti.Vector.field(3, ti.f32, shape=4)
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
    # cube
    X[0] = 0.5, 0.5, 0.5 ;M[0] = 1.
    X[1] = 0.45, 0.4, 0.45 ;M[1] = 1.
    X[2] = 0.55, 0.4, 0.45 ;M[2] = 1.
    X[3] = 0.5, 0.4, 0.55 ;M[3] = 1.

    initP()

    #floor
    X[4]  = 0., 0., 0. ;M[4] = 0.
    X[5]  = 0., 0., 1. ;M[5] = 0.
    X[6] = 1., 0., 1.  ;M[6] = 0.
    X[7] = 1., 0., 0.  ;M[7] = 0.

#
# Volume Conservation
#

def solve():
    pass
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
    
gui = ti.GUI('MPM3D', background_color=0x112F41)
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
        K[None] *= 1.01
    if gui.is_pressed('q'):
        K[None] /= 1.01


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