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
points = []
edges  =[[], []]
cloth1 = RectClothMesh(createBowCloth(25,25,0.02, 0.02, .2, (.25, .25, .5)), 0)
cloth2 = RectClothMesh(createBowCloth(25,25,0.02, 0.02, .05,(.25, .25, .5), -1), len(cloth1.points))
points.extend(cloth1.points); points.extend(cloth2.points)
edges[0].extend(cloth1.edges[0]); edges[1].extend(cloth1.edges[1])
edges[0].extend(cloth2.edges[0]); edges[1].extend(cloth2.edges[1])
# calc cm
cm = np.zeros(3)
for p in points:
    cm += np.array(p)
points.append(cm/len(points)+[.12,.12,.0])
points.append(cm/len(points)-[.12,.12,.0])

N = len(points) + 4

memory = Memory(N)
# add floor
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])
# camera
camera = Camera(focus=(.5, .5,.5), angle=(0., 0.), scale=.8)
# solvers
stretchSolver = TotalStretchSolver(memory, N, len(edges[0])-5, restStiff=.5)
shmSolver     = ShapeMatchingSolver(memory, N)

sewPoints = set()
for i in range(25):
    x = 24*25+i
    sewPoints.add(x);sewPoints.add(625+x)
    x = i
    sewPoints.add(x);sewPoints.add(625+x)
    x = 25*i+24
    sewPoints.add(x);sewPoints.add(625+x)
    x = 25*i
    sewPoints.add(x);sewPoints.add(625+x)
sewSolver     = TotalSewSolver(    memory, N, len(sewPoints)//2)


cells = Delaunay(points).simplices
volSolver     = VolumeSolver(memory, len(cells))

# pbd
pbd         = PostionBasedDynamics(memory, camera, N, restIter=15)


def init():
    pbd.reset()
    # reset solvers
    stretchSolver.reset()
    sewSolver.reset()
    shmSolver.reset()
    volSolver.reset()
    # mesh
    for i in range(len(points)):
        memory.update(i, points[i], 1.)
        shmSolver.update(i, i)
    shmSolver.init()

       
    for i in range(len(edges[0])-5):
         stretchSolver.update(i, edges[0][i], edges[1][i])
    stretchSolver.init()

    for i,x in enumerate([x for x in sewPoints if x < 625]):
        sewSolver.update(i, x, x+625)
    sewSolver.init() 

    for i,(x,y,z,w) in enumerate(cells):
        volSolver.update(i, x,y,z,w)
    volSolver.init() 
    
    #floor
    memory.update(N-4, [0., 0., 0.], 0.)
    memory.update(N-3, [0., 0., 1.], 0.)
    memory.update(N-2, [1., 0., 1.], 0.)
    memory.update(N-1, [1., 0., 0.], 0.)


def step(paused, mouse_pos, picked):
    if not paused:
        for _ in range(pbd.substep):
            pbd.apply_force(mouse_pos[0], mouse_pos[1], picked)
            shmSolver.solve()
            for _ in range(pbd.iters[None]):
                volSolver.solve()
                #stretchSolver.solve()
                sewSolver.solve()
            pbd.update()
            pbd.floor_confinement()

def render(gui, pos2):
    # render
    scale = camera.getScale()
    gui.circles(pos2, radius=1*scale, color=0x66ccff)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xffeedd, radius=.2*scale)
    gui.text(content=f'Stiffness={volSolver.K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iteration={pbd.iters[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Shape={shmSolver.ALPHA[None]}',pos=(0,0.85), color=0xffffff)
    gui.text(content=f'Gravity={pbd.gravity[None].value}',pos=(0,0.8), color=0xffffff)


    
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

    eventHandler.regularReact(camera=camera, stiffness_field=volSolver.K, iteration_field=pbd.iters, mass_field=memory.invM,
                            pos2d=pos2,
                            init_method=init, step_method=step, pick_method=pick
                            )
    eventHandler.fastReact(camera=camera, 
                        pos2d=pos2,
                        pick_method=pick)
    if gui.is_pressed(","):
        shmSolver.ALPHA[None] /= 1.1
    elif gui.is_pressed('.'):
        shmSolver.ALPHA[None] *= 1.1
    if gui.is_pressed("="):
        pbd.gravity[None][1] /= 1.1
    elif gui.is_pressed('-'):
        pbd.gravity[None][1] *= 1.1
    
    render(gui, pos2)
    gui.show()
    # sim
    step(eventHandler.paused, eventHandler.mouse_pos, eventHandler.picked)