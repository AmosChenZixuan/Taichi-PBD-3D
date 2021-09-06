'''
    https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf
'''
import taichi as ti # 0.7.29
import numpy as np
from numpy.linalg import norm
import pygalmesh
from scipy.spatial.transform import Rotation

from include import *
from solvers import *
from UI import Camera, EventHandler
ti.init(arch=ti.gpu)

# build scene objects
s = pygalmesh.Stretch(pygalmesh.Ball([0, 0, 0], 1.0), [1.0, 2.0, 0.0])
mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=0.8, verbose=False)
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
memory = Memory(N)
# add floor
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])
# camera
camera = Camera(focus=(.5, .5,.5), angle=(5., 1.), scale=.8)
# volume 
NC            = len(triangulation)           # number of constraint
tetSolver = TetrahedronSolver(memory, N-4, NC)
# shape
shapeSolver = ShapeMatchingSolver(memory, N-4)
# pbd
GRAVITY = ti.Vector.field(3, ti.f32, shape=())
GRAVITY[None] = ti.Vector([0., -9.8, 0.])
DeltaT  = 1 / 240
iter    = ti.field(ti.i32, shape=())   # solver iterations
iter[None] = 3



def init():
    # reset solvers
    tetSolver.reset()
    shapeSolver.reset()

    # mesh
    offset = 0
    for i in range(N-4):
        memory.update(i, points[i], 1.)
        shapeSolver.update(i, offset+i)
    shapeSolver.init()

    # constraint
    for i in range(NC):
        x,y,z,w = triangulation[i]
        tetSolver.update(i, x,y,z,w)
    tetSolver.init()
    
    #floor
    memory.update(N-4, [0., 0., 0.], 0.)
    memory.update(N-3, [0., 0., 1.], 0.)
    memory.update(N-2, [1., 0., 1.], 0.)
    memory.update(N-1, [1., 0., 0.], 0.)

    
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
        if memory.invM[i] <= 0.: continue
        memory.vel[i] += GRAVITY[None] * DeltaT
        memory.newPos[i]  = memory.curPos[i] + memory.vel[i] * DeltaT

        # mouse interaction
        if idx>=0:
            memory.newPos[idx] = mouse_x, mouse_y, memory.newPos[idx][2]

@ti.kernel
def update():
    for i in range(N):
        if memory.invM[i] <= 0.: continue
        memory.vel[i] = (memory.newPos[i] - memory.curPos[i]) / DeltaT * .99
        memory.curPos[i] = memory.newPos[i]

@ti.kernel
def box_confinement():
    for i in range(N):
        if memory.newPos[i][1] < 0.:
            memory.newPos[i][1] = 1e-4

def step(paused, mouse_pos, picked):
    if not paused:
        apply_force(mouse_pos[0], mouse_pos[1], picked)
        box_confinement()
        for _ in range(iter[None]):
            tetSolver.solve()
        shapeSolver.solve()
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
    pos3 = memory.curPos.to_numpy()
    pos2 = np.zeros((len(pos3), 2))
    project(pos3, pos2)

    eventHandler.regularReact(camera=camera, stiffness_field=tetSolver.K, iteration_field=iter, mass_field=memory.invM,
                            pos2d=pos2,
                            init_method=init, step_method=step, pick_method=pick
                            )
    eventHandler.fastReact(camera=camera, 
                        pos2d=pos2,
                        pick_method=pick)
    if gui.is_pressed(","):
        shapeSolver.ALPHA[None] /= 1.1
    elif gui.is_pressed('.'):
        shapeSolver.ALPHA[None] *= 1.1
    if gui.is_pressed("="):
        GRAVITY[None][1] /= 1.1
    elif gui.is_pressed('-'):
        GRAVITY[None][1] *= 1.1
    
    # render
    scale = camera.getScale()
    #gui.circles(pos2, radius=2*scale, color=0x66ccff)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xffeedd, radius=.5*scale)
    gui.text(content=f'Stiffness={tetSolver.K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={iter[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Shape={shapeSolver.ALPHA[None]}',pos=(0,0.85), color=0xffffff)
    gui.text(content=f'Gravity={GRAVITY[None].value}',pos=(0,0.8), color=0xffffff)
    gui.show()
    # sim
    step(eventHandler.paused, eventHandler.mouse_pos, eventHandler.picked)