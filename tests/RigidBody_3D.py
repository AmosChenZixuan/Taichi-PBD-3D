import taichi as ti # 0.7.29
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, norm
from scipy.spatial.transform import Rotation
import time
from include.utils import timeThis

ti.init(arch=ti.gpu)

X    = ti.Vector.field(3, ti.f32, shape=12)
P    = ti.Vector.field(3, ti.f32, shape=12)
V    = ti.Vector.field(3, ti.f32, shape=12)
view = ti.Vector.field(2, ti.f32, shape=14)
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
Aqq = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())  # not used
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
    p[None] = 5.
    t[None] = 15.
    focus[None] = [0.5, 0.5, 0.5]
    scale[None] = 0.8

def init():
    V.fill(0)
    P.fill(0)
    Apq.fill(0)
    R.fill(0)
    # cube
    X[0] = 0.4, 0.6, 0.4
    X[1] = 0.4, 0.6, 0.6
    X[2] = 0.6, 0.6, 0.6
    X[3] = 0.6, 0.6, 0.4
    X[4] = 0.3, 0.4, 0.4
    X[5] = 0.3, 0.4, 0.6
    X[6] = 0.6, 0.5, 0.6
    X[7] = 0.6, 0.5, 0.4
    initP()
    updateCM()
    updateQ()
    initQ0()
    calcApq()
    #initAqq()
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

def initAqq():
    A = Apq[None].value.to_numpy()
    Aqq[None] = ti.Vector(inv(A))

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
        # A[0,0] += Q0[i][0] * Q[i][0]
        # A[1,0] += Q0[i][0] * Q[i][1]
        # A[2,0] += Q0[i][0] * Q[i][2]

        # A[0,1] += Q0[i][1] * Q[i][0]
        # A[1,1] += Q0[i][1] * Q[i][1]
        # A[2,1] += Q0[i][1] * Q[i][2]

        # A[0,2] += Q0[i][2] * Q[i][0]
        # A[1,2] += Q0[i][2] * Q[i][1]
        # A[2,2] += Q0[i][2] * Q[i][2]
    Apq[None] = A

@timeThis
def calcR_polar():
    # R = AS^-1     S = AA^T ** .5
    try:
        A = Apq[None].value.to_numpy()
        S = sqrtm(A.T@A)
        R[None] = ti.Vector(A @ inv(S))
    except:
        R[None] = ti.Vector(quaternion(np.array([0., 0., 0., 1.])))

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

@timeThis
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
    for i in ti.static(range(8)):
        p = R[None] @ Q0[i] + CM[None]
        P[i] += (p - P[i]) * ALPHA[None] 

def solve():
    updateCM()
    updateQ()
    calcApq()
    if method:
        calcR_extract()
    else:
        calcR_polar()
    updateDelta()

#
# PBD
#
@ti.kernel
def apply_force(mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
    for i in ti.static(range(8)):
        V[i] += GRAVITY * DeltaT
        P[i]  = X[i] + V[i] * DeltaT

        # mouse interaction
        if attract:
            P[0] = mouse_x, mouse_y, P[0][2]

@ti.kernel
def update():
    for i in ti.static(range(8)):
        V[i] = (P[i] - X[i]) / DeltaT * .99
        X[i] = P[i]
    # pinned
    X[7] = 0.6, 0.5, 0.4
    P[7] = 0.6, 0.5, 0.4
    V[7] = 0., 0., 0.

@ti.kernel
def box_confinement():
    for i in ti.static(range(8)):
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
    view[12] = ti.Vector([u,v]) * scale[None] + ti.Vector([0.5, 0.5])
    # cm
    x,y,z = CM[None] - focus[None]
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    view[13] = ti.Vector([u,v]) * scale[None] + ti.Vector([0.5, 0.5])

gui = ti.GUI('MPM3D', background_color=0x112F41)
initCamera()
init()

method = True
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
        ALPHA[None] *= 1.01
    if gui.is_pressed('q'):
        ALPHA[None] /= 1.01
    if gui.is_pressed('z'):
        method = not method
        time.sleep(0.3)
        print(f"RigidSolver: {'RotExt' if method else 'Polar'}")

    if gui.is_pressed(ti.GUI.RMB):
        mouse_pos = gui.get_cursor_pos()
        attract = 1
    elif gui.is_pressed(ti.GUI.LMB):
        mouse_pos = gui.get_cursor_pos()
        attract = -1
    else:
        attract = 0

    
    gui.circles(pos[:12], radius=5*scale[None], color=0x66ccff)
    gui.circle(pos[12], radius=1*scale[None], color=0xff0000)
    gui.circle(pos[13], radius=10*scale[None], color=0xff0000)

    gui.lines(pos[edges[0]], pos[edges[1]], color=0x66ccff, radius=1*scale[None])
    gui.show()

    step()