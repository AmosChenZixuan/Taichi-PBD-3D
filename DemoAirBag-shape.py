import taichi as ti # 0.7.29
import numpy as np
from scipy.spatial import Delaunay

from include import *
from solvers import * 
from UI import Camera, EventHandler
ti.init(arch=ti.gpu)

class RectClothMesh:
    def __init__(self, vert, offset):
        self.points = vert
        points2d = np.array([flatten(x) for x in vert])
        self.cells  = Delaunay(points2d).simplices
        self.edges=[[], []]
        for cell in self.cells:
            for i in range(3):
                for j in range(i+1, 3):
                    self.edges[0].append(cell[i]+offset)
                    self.edges[1].append(cell[j]+offset)

# build scene objects
edges=[[], []]
cloth1 = RectClothMesh(createRectCloth(25,25,0.02, 0.02, (.25, .25, .4)), 0)
cloth2 = RectClothMesh(createRectCloth(25,25,0.02, 0.02, (.25, .25, .6)), len(cloth1.points))
edges[0].extend(cloth1.edges[0]); edges[1].extend(cloth1.edges[1])
edges[0].extend(cloth2.edges[0]); edges[1].extend(cloth2.edges[1])

N = len(cloth1.points) + len(cloth2.points) + 4

memory = Memory(N)
# add floor
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])
# camera
camera = Camera(focus=(.5, .5,.5), angle=(0., 0.), scale=.8)
# solvers
stretchSolver = TotalStretchSolver(memory, N, len(edges[0])-5)
sewSolver     = TotalSewSolver(    memory, N, 100)
shmSolver     = ShapeMatchingSolver(memory, N)
# pbd
pbd         = PostionBasedDynamics(memory, camera, N, restIter=15)


def init():
    pbd.reset()
    # reset solvers
    stretchSolver.reset()
    sewSolver.reset()
    shmSolver.reset()
    # mesh
    for i in range(len(cloth1.points)):
        memory.update(i, cloth1.points[i], 1.)
        shmSolver.update(i, i)
    offset = len(cloth1.points)
    for i in range(len(cloth2.points)):
        memory.update(i+offset, cloth2.points[i], 1.)
        shmSolver.update(i+offset, i+offset)
    shmSolver.init()

       
    for i in range(len(edges[0])-5):
         stretchSolver.update(i, edges[0][i], edges[1][i])
    stretchSolver.init()

    for i in range(25):
        sewSolver.update(i, 24*25+i, 625+24*25+i)
    for i in range(25):
        sewSolver.update(25+i, i, 25*25+i)
    for i in range(25):
        sewSolver.update(50+i, 25*i+24, 25*25+25*i+24)
    for i in range(25):
        sewSolver.update(75+i, 25*i, 25*25+25*i)  
    sewSolver.init()  
    
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
            shmSolver.solve()
            for _ in range(pbd.iters[None]):
                stretchSolver.solve()
                sewSolver.solve()
                
            pbd.update()

def render(gui, pos2):
    # render
    scale = camera.getScale()
    gui.circles(pos2, radius=2*scale, color=0x66ccff)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xffeedd, radius=.5*scale)
    gui.text(content=f'Stiffness={shmSolver.ALPHA[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={pbd.iters[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Gravity={pbd.gravity[None].value}',pos=(0,0.85), color=0xffffff)


    
###
###
###
gui = ti.GUI('Airbag', background_color=0x112F41)
eventHandler = EventHandler(gui)
eventHandler.paused = True
init()
while gui.running:
    pos3 = memory.curPos.to_numpy()
    pos2 = np.zeros((len(pos3), 2))
    pbd.project(pos3, pos2)

    eventHandler.regularReact(camera=camera, stiffness_field=shmSolver.ALPHA, iteration_field=pbd.iters, mass_field=memory.invM,
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