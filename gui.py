import taichi as ti

ti.init(arch = ti.gpu)

gui = ti.GUI('Window', (1024, 768))

s = (0.0)
go = False
while gui.running:
    for e in gui.get_events():
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()

        if e.key == ti.GUI.WHEEL:
            print(e.pos, e.delta.x, e.delta.y)

        if e.key == ti.GUI.LMB:
            if e.type==ti.GUI.PRESS:
                go = True
                s = e.pos
            elif e.type==ti.GUI.RELEASE:
                go = False
        if e.type==ti.GUI.MOTION and go:
            #print(ti.Vector([e.pos[0]-s[0], e.pos[1]-s[1]]).normalized())
            print((ti.Vector(e.pos)-ti.Vector(s)).normalized())
            s = e.pos


    gui.show()