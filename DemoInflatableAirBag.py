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
        self.edges= getEdges(self.cells, offset)

# build scene objects
points = []
edges  =[[], []]
# cloth1 = RectClothMesh(createSemiSphereCloth(25,25,0.02, 0.02, .1, (.25, .25, .5), -1), 0)
# cloth2 = RectClothMesh(createSemiSphereCloth(25,25,0.02, 0.02, .1, (.25, .25, .55)  ), len(cloth1.points))
cloth1 = RectClothMesh(createRectCloth(25,25,0.02, 0.02, (.25, .25, .5)), 0)
cloth2 = RectClothMesh(createRectCloth(25,25,0.02, 0.02, (.25, .25, .51)), len(cloth1.points))
points.extend(cloth1.points); points.extend(cloth2.points)
edges[0].extend(cloth1.edges[0]); edges[1].extend(cloth1.edges[1])
edges[0].extend(cloth2.edges[0]); edges[1].extend(cloth2.edges[1])


N = len(points) + 4

memory = Memory(N)
# camera
camera = Camera(focus=(.5, .5,.5), angle=(0., 0.), scale=1.)
# solvers
stretchSolver = TotalStretchSolver(memory, N, len(edges[0])-5, restStiff=.5)
shmSolver     = ShapeMatchingSolver(memory, N, .1)

sewPoints = set()
cells = []
for i in range(25):
    x = 24*25+i
    sewPoints.add(x);sewPoints.add(625+x)
    x = i
    sewPoints.add(x);sewPoints.add(625+x)
    x = 25*i+24
    sewPoints.add(x);sewPoints.add(625+x)
    x = 25*i
    sewPoints.add(x);sewPoints.add(625+x)
    x = 8*25+i
    sewPoints.add(x);sewPoints.add(625+x)
    x = 16*25+i
    sewPoints.add(x);sewPoints.add(625+x)
    # if i < 24:
    #     cells.extend([
    #         (24*25+i, 24*25+(i+1), 24*25+i+625), (24*25+i, 24*25+(i+1), 24*25+(i+1)+625),
    #         (i, i+1, i+625), (i, i+1, i+1+625),
    #         (25*i+24, 25*(i+1)+24, 25*i+24+625), (25*i+24, 25*(i+1)+24, 25*(i+1)+24+625), 
    #         (25*i, 25*(i+1), 25*i+625), (25*i, 25*(i+1), 25*(i+1)+625)
    #     ])
sewSolver     = TotalSewSolver(    memory, N, len(sewPoints)//2)

cells.extend(list(cloth1.cells))
offset = len(cloth1.points)
for c in cloth2.cells:
    x,y,z = c
    cells.append((z+offset, y+offset, x+offset))
volSolver     = ClothBalloonSolver(memory, len(points), len(cells), 1.)
# pbd
pbd         = PostionBasedDynamics(memory, camera, N, restIter=5)


# add floor
edges = getEdges(cells)
edges[0].extend([N-4,N-3,N-2,N-1, N-4])
edges[1].extend([N-3,N-2,N-1,N-4, N-2])


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

    for i, vert in enumerate(cells):
        volSolver.update(i, *vert)
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
                stretchSolver.solve()
                sewSolver.solve()
                volSolver.solve()
                
            pbd.update()
            pbd.floor_confinement()
            pbd.ceiling_confinement()

def render(gui, pos2):
    # render
    scale = camera.getScale()
    gui.circles(pos2, radius=1*scale, color=0x66ccff)
    #gui.circle(camera.project(camera.getFocus()), radius=1*scale, color=0xff0000)
    gui.line([0, pbd.ceil[None]], [1, pbd.ceil[None]],color=0xffeedd, radius=2*scale )
    gui.lines(pos2[edges[0]], pos2[edges[1]], color=0xffeedd, radius=.2*scale)
    gui.text(content=f'Pressure={volSolver.K[None]}',pos=(0,0.95), color=0xffffff)
    gui.text(content=f'Iterations={pbd.iters[None]}',pos=(0,0.9), color=0xffffff)
    gui.text(content=f'Shape={shmSolver.ALPHA[None]}',pos=(0,0.85), color=0xffffff)
    gui.text(content=f'Ceiling={pbd.ceil[None]}',pos=(0,0.8), color=0xffffff)


    
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
        pbd.ceil[None] = min(pbd.ceil[None]+.01, 1.)
    elif gui.is_pressed('-'):
        pbd.ceil[None] = max(pbd.ceil[None]-.01, .01)
    elif gui.is_pressed('0'):
        pbd.ceil[None] = 1.
    
    render(gui, pos2)
    gui.show()
    # sim
    step(eventHandler.paused, eventHandler.mouse_pos, eventHandler.picked)