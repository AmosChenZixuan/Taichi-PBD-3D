import taichi as ti # 0.8.1
import numpy as np
from scipy.spatial import Delaunay

from include import *
from solvers import *   
ti.init(arch=ti.gpu)

class RectClothMesh:
    def __init__(self, vert, offset):
        self.points = vert
        points2d = np.array([[x[0],x[1]] for x in vert])
        self.cells  = Delaunay(points2d).simplices
        self.edges= getEdges(self.cells, offset)

# build scene objects
points = []
edges  =[[], []]
cloth1 = RectClothMesh(createRectCloth(25,25,0.02, 0.02, (.25, .25, .5)), 0)
cloth2 = RectClothMesh(createRectCloth(25,25,0.02, 0.02, (.25, .25, .51)), len(cloth1.points))
points.extend(cloth1.points); points.extend(cloth2.points)
edges[0].extend(cloth1.edges[0]); edges[1].extend(cloth1.edges[1])
edges[0].extend(cloth2.edges[0]); edges[1].extend(cloth2.edges[1])


N = len(points) + 8 
memory = Memory(N)
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
    # x = 8*25+i
    # sewPoints.add(x);sewPoints.add(625+x)
    # x = 16*25+i
    # sewPoints.add(x);sewPoints.add(625+x)
sewSolver     = TotalSewSolver(    memory, N, len(sewPoints)//2)

cells.extend(list(cloth1.cells))
offset = len(cloth1.points)
for c in cloth2.cells:
    x,y,z = c
    cells.append((z+offset, y+offset, x+offset))
volSolver     = ClothBalloonSolver(memory, len(points), len(cells), 1.)
# pbd
pbd         = PostionBasedDynamics(memory, nParticle = N, restIter=5)


# add floor
edges = getEdges(cells)
edges[0].extend([N-4,N-3,N-2,N-1, N-4, N-8,N-7,N-6,N-5, N-8])
edges[1].extend([N-3,N-2,N-1,N-4, N-2, N-7,N-6,N-5,N-8, N-6])

# mesh indices
meshIdx = field(len(cells)*3, 1, ti.i32)
for i in range(len(cells)):
    meshIdx[i*3+0] = cells[i][0]
    meshIdx[i*3+1] = cells[i][1]
    meshIdx[i*3+2] = cells[i][2]

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
    memory.update(N-8, [0., 1., 0.], 0.)
    memory.update(N-7, [0., 1., 1.], 0.)
    memory.update(N-6, [1., 1., 1.], 0.)
    memory.update(N-5, [1., 1., 0.], 0.)
    memory.update(N-4, [0., 0., 0.], 0.)
    memory.update(N-3, [0., 0., 1.], 0.)
    memory.update(N-2, [1., 0., 1.], 0.)
    memory.update(N-1, [1., 0., 0.], 0.)

def step(paused, mouse_pos, picked):
    if not paused:
        for _ in range(pbd.substep):
            pbd.apply_force(mouse_pos[0], mouse_pos[1], picked)
            #shmSolver.solve()
            for _ in range(pbd.iters[None]):
                stretchSolver.solve()
                sewSolver.solve()
                volSolver.solve()
                
            pbd.update()
            pbd.floor_confinement()
            pbd.ceiling_confinement()
    for i in range(5, 9):
        memory.curPos[N-i][1] = pbd.ceil[None]

init()
res = (1920, 1080)
window = ti.ui.Window("AirBag 3D", res, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((.1, .3, .4))
scene  = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)

ticks   = 0
paused  = False
pRadius = .01
pColor  = (.9,.8,.7)
iters   = pbd.iters[None]
def show_options():
    global paused, pRadius, iters

    window.GUI.begin("Debuug", 0.05, 0.1, 0.2, 0.5)
    volSolver.K[None] = window.GUI.slider_float("Stiffness", volSolver.K[None], 0, 100)

    #pbd.gravity[None][0] = window.GUI.slider_float("x", pbd.gravity[None][0], -10, 10)
    pbd.gravity[None][1] = window.GUI.slider_float("Gravity-y", pbd.gravity[None][1], -10, 10)
    #pbd.gravity[None][2] = window.GUI.slider_float("z", pbd.gravity[None][2], -10, 10)

    iters = window.GUI.slider_float("Iteration", iters, 0, 20)
    pbd.iters[None] = int(iters)

    pbd.ceil[None] = window.GUI.slider_float("Ceiling", pbd.ceil[None], 0, 1)
    pRadius = window.GUI.slider_float("particles radius ", pRadius, 0, 0.1)

    if window.GUI.button("restart"):
        init()
    if paused:
        if window.GUI.button("Continue"):
            paused = False
    else:
        if window.GUI.button("Pause"):
            paused = True
    window.GUI.end()

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.7, 0.7, 0.7))

    scene.mesh(memory.curPos, meshIdx, color=pColor)

    scene.point_light(pos=(0.5, 1.5, -1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5,  1.5), color=(0.5, 0.5, 0.5))
    canvas.scene(scene)

while window.running:
    #print("heyyy ",frame_id)
    ticks += 1
    if not paused:
        step(paused, (0,0), -1)

    render()

    show_options()
    try:
        window.show()
    except RuntimeError:
        import traceback
        traceback.print_exc()