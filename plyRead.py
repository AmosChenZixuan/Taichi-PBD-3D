import taichi as ti
import numpy as np
from plyfile import PlyData, PlyElement

ti.init(arch=ti.gpu)



plydata = PlyData.read('silica.ply')

N  = len(plydata['vertex'])
NT = len(plydata['face'])

X = ti.Vector.field(3, ti.f32, N)
C = ti.Vector.field(3, ti.f32, N)
T = ti.Vector.field(3, ti.f32, NT)


for i in range(N):
    X[i] = plydata.elements[0].data[i][0], plydata.elements[0].data[i][1], plydata.elements[0].data[i][2]

    C[i] = plydata.elements[0].data[i][3]/255, plydata.elements[0].data[i][4]/255, plydata.elements[0].data[i][5]/255


res = (1920, 1080)
window = ti.ui.Window("AirBag 3D", res, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((.1, .3, .4))
scene  = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.7, 0.7, 0.7))

    scene.particles(X, .1, per_vertex_color=C)

    scene.point_light(pos=(0.5, 1.5, -1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5,  1.5), color=(0.5, 0.5, 0.5))
    canvas.scene(scene)

while window.running:
    render()

    try:
        window.show()
    except RuntimeError:
        import traceback
        traceback.print_exc()