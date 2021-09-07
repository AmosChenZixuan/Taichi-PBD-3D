import taichi as ti # 0.7.29
import numpy as np
from scipy.spatial import Delaunay

from include import *
from solvers import *
from UI import Camera, EventHandler
ti.init(arch=ti.gpu)


# build scene objects
leftbtm = (.25, .25, .5)
points = []
np.random.seed(0)
points.extend(createRectCloth(25,25,0.02, 0.02, leftbtm))
flatten  = Camera(focus=(.5,.5,.5)).project
points2d = np.array([flatten(x) for x in points])
triangulation = Delaunay(points2d)
edges=[[], []]
for tet in triangulation.simplices:
    for i in range(3):
        for j in range(i+1, 3):
            edges[0].append(tet[i])
            edges[1].append(tet[j])

N = len(points) + 4

memory = Memory(N)
# add floor
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])
# camera
camera = Camera(focus=(.5, .5,.5), angle=(5., 1.), scale=.8)
# strech
stretchSolver = TotalStretchSolver(memory, N-4, len(edges[0])-5)
# pbd
pbd         = PostionBasedDynamics(memory, camera, N)


def init():
    pbd.reset()
    # reset solvers
    stretchSolver.reset()

    # mesh
    for i in range(N-4):
        memory.update(i, points[i], 1.)
    #memory.invM[0] = 0.

    for i in range(len(edges[0])-5):
        stretchSolver.update(i, edges[0][i], edges[1][i])
    stretchSolver.init()

    
    #floor
    memory.update(N-4, [0., 0., 0.], 0.)
    memory.update(N-3, [0., 0., 1.], 0.)
    memory.update(N-2, [1., 0., 1.], 0.)
    memory.update(N-1, [1., 0., 0.], 0.)


def step(paused, mouse_pos, picked):
    if not paused:
        for _ in range(pbd.substep):
            pbd.apply_force(mouse_pos[0], mouse_pos[1], picked)
            pbd.box_confinement()
            for _ in range(pbd.iters[None]):
                stretchSolver.solve()
            pbd.update()

def render(gui, pos2):
    # render
    scale = camera.getScale()
    gui.circles(pos2, radius=2*scale, color=0x66ccff)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xffeedd, radius=.5*scale)
    gui.text(content=f'Stiffness={stretchSolver.K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={pbd.iters[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Gravity={pbd.gravity[None].value}',pos=(0,0.85), color=0xffffff)


    
###
###
###
gui = ti.GUI('Cloth', background_color=0x112F41)
eventHandler = EventHandler(gui)
init()
while gui.running:
    pos3 = memory.curPos.to_numpy()
    pos2 = np.zeros((len(pos3), 2))
    pbd.project(pos3, pos2)

    eventHandler.regularReact(camera=camera, stiffness_field=stretchSolver.K, iteration_field=pbd.iters, mass_field=memory.invM,
                            pos2d=pos2,
                            init_method=init, step_method=step, pick_method=pick
                            )
    eventHandler.fastReact(camera=camera, 
                        pos2d=pos2,
                        pick_method=pick)
    if gui.is_pressed("="):
        pbd.gravity[None][1] /= 1.1
    elif gui.is_pressed('-'):
        pbd.gravity[None][1] *= 1.1
    
    render(gui, pos2)
    gui.show()
    # sim
    step(eventHandler.paused, eventHandler.mouse_pos, eventHandler.picked)