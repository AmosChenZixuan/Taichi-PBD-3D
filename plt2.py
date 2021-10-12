import taichi as ti
import numpy as np
from plyfile import PlyData

ti.init(arch=ti.gpu)

def load_mesh(fn, scale=1, offset=(0, 0, 0)):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    r = plydata['vertex']['red']
    g = plydata['vertex']['green']
    b = plydata['vertex']['blue']
    return x, y, z, r,g,b
    # elements = plydata['face']
    # num_tris = len(elements['vertex_indices'])
    # triangles = np.zeros((num_tris, 9), dtype=np.float32)

    # for i, face in enumerate(elements['vertex_indices']):
    #     assert len(face) == 3
    #     for d in range(3):
    #         triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
    #         triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
    #         triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    # return triangles

x,y,z,r,g,b = load_mesh('silica.ply')

X = ti.Vector.field(3, ti.f32, len(x))
C = ti.Vector.field(3, ti.f32, len(r))

r = np.array(r)/255
g = np.array(g)/255
b = np.array(b)/255

@ti.kernel
def init(x:ti.ext_arr(),y:ti.ext_arr(),z:ti.ext_arr(),r:ti.ext_arr(),g:ti.ext_arr(),b:ti.ext_arr(), n:int):
    for i in range(n):
        X[i] = x[i], y[i], z[i]
        C[i] = r[i], g[i], b[i]
        
    

init(x,y,z,r,g,b, len(x))


res = (1920, 1080)
window = ti.ui.Window("PLY 3D", res, vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((.1, .3, .4))
scene  = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)
pRadius = .003

def show_options():
    global pRadius
    window.GUI.begin("Debuug", 0.05, 0.1, 0.2, 0.1)
    pRadius = window.GUI.slider_float("particles radius ", pRadius, 0, 0.1)
    window.GUI.end()

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.7, 0.7, 0.7))

    scene.particles(X, pRadius, per_vertex_color=C)

    scene.point_light(pos=(0.5, 1.5, -1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5,  1.5), color=(0.5, 0.5, 0.5))
    canvas.scene(scene)

while window.running:
    render()
    show_options()
    try:
        window.show()
    except RuntimeError:
        import traceback
        traceback.print_exc()