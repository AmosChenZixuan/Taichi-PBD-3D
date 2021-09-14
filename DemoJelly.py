import taichi as ti # 0.7.29
import numpy as np
from scipy.spatial import Delaunay

from include import *
from solvers import *
from UI import Camera, EventHandler
ti.init(arch=ti.gpu)


# build scene objects
points, cells = createGalCube((.35,.35,.35), 30)
N = len(points)
triangulation = cells[1].data
edges         = getEdges(triangulation)
surfaceVert   = getSurfaceVertices(cells[0].data)
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
tetSolver     = VolumeSolver(memory, N-4, NC, .8)
tetSolver2     = TetrahedronSolver(memory, N-4, NC)
# shape
stretchSolver = TotalStretchSolver( memory, N, len(edges[0])-5, restStiff=1.)
shapeSolver   = ShapeMatchingSolver(memory, N-4, .2)
# pbd
pbd         = PostionBasedDynamics(memory, camera, N)


def init():
    pbd.reset()
    # reset solvers
    tetSolver.reset()
    tetSolver2.reset()
    shapeSolver.reset()

    # mesh
    for i in range(N-4):
        memory.update(i, points[i], 1.)
        shapeSolver.update(i, i)
    # for i,j in enumerate(surfaceVert):
    #     shapeSolver.update(i, j)
    shapeSolver.init()

    # constraint
    for i in range(NC):
        x,y,z,w = triangulation[i]
        tetSolver.update(i, x,y,z,w)
        tetSolver2.update(i, x,y,z,w)
    tetSolver.init()
    tetSolver2.init()

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
            #shapeSolver.solve()
            for _ in range(pbd.iters[None]):
                tetSolver.solve()
                tetSolver2.solve()
                stretchSolver.solve()
            pbd.update()
            pbd.floor_confinement()
            pbd.ceiling_confinement()


###
###
###
gui = ti.GUI('SoftBall', res=(1000,1000), background_color=0x112F41)
eventHandler = EventHandler(gui)
init()
while gui.running:
    pos3 = memory.curPos.to_numpy()
    pos2 = np.zeros((len(pos3), 2))
    pbd.project(pos3, pos2)

    eventHandler.regularReact(camera=camera, stiffness_field=tetSolver.K, iteration_field=pbd.iters, mass_field=memory.invM,
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
        pbd.ceil[None] = min(pbd.ceil[None]+.01, 1.)
    elif gui.is_pressed('-'):
        pbd.ceil[None] = max(pbd.ceil[None]-.01, .01)
    
    # render
    scale = camera.getScale()
    gui.circles(pos2, radius=2*scale, color=0x66ccff)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xffeedd, radius=.5*scale)
    gui.line([0, pbd.ceil[None]], [1, pbd.ceil[None]],color=0xffeedd, radius=2*scale )
    gui.text(content=f'Stiffness={tetSolver.K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={pbd.iters[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Shape={shapeSolver.ALPHA[None]}',pos=(0,0.85), color=0xffffff)
    gui.text(content=f'Ceiling={pbd.ceil[None]}',pos=(0,0.8), color=0xffffff)
    gui.show()
    # sim
    step(eventHandler.paused, eventHandler.mouse_pos, eventHandler.picked)