import taichi as ti


class EventHandler:
    def __init__(self, gui:ti.GUI):
        self.gui = gui

        self.paused  = False
        self.picked  = -1
        self.mouse_pos = (0, 0)
        self.RMB     = False
    
    def regularPress(self, e, *args, **kargs):
        gui = self.gui
        init   = kargs['init_method']
        camera = kargs['camera']
        step   = kargs['step_method']
        K      = kargs['stiffness_field']
        niter  = kargs['iteration_field']
        if e.key == gui.SPACE:
            self.paused = not self.paused
        elif e.key == 'r':
            init()
        elif e.key == 'c':
            camera.reset()
        elif e.key == 't':
            self.paused = False
            step(self.paused, self.mouse_pos, self.picked)
            self.paused = True
        # stiffness
        elif e.key == 'e':
            K[None] *= 1.05
        elif e.key == 'q':
            K[None] /= 1.05
        # iteration
        if gui.is_pressed(']'):
            niter[None] += 1
        elif gui.is_pressed('['):
            niter[None] -= 1
        # start rotating
        if e.key == ti.GUI.RMB:
            self.RMB = True
            self.mouse_pos = e.pos

    def regularRelease(self, e, *args, **kargs):
        # exit rotating
        if e.key == ti.GUI.RMB:
            self.RMB = False

    def regularMotion(self, e, *args, **kargs):
        camera = kargs['camera']
        # rotating
        if self.RMB:
            xdelta, ydelta = (ti.Vector(e.pos)-ti.Vector(self.mouse_pos)) * 100
            camera.rotate(xdelta, ydelta)
            self.mouse_pos = e.pos
        # scale/zoom
        if e.key == ti.GUI.WHEEL:
            camera.zoom(1.01 if e.delta.y>0 else 1/1.01)


    def regularReact(self, *args, **kargs):
        gui = self.gui
        for e in gui.get_events():
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            if e.type == gui.PRESS:
                self.regularPress(e, *args, **kargs)
            elif e.type == gui.RELEASE:
                self.regularRelease(e, *args, **kargs)
            else:
                self.regularMotion(e, *args, **kargs)

    def fastReact(self, *args, **kargs):
        gui = self.gui
        camera = kargs['camera']
        pick   = kargs['pick_method']
        pos2   = kargs['pos2d']
        # camera angle
        if gui.is_pressed(ti.GUI.LEFT):
            camera.rotate(-1., 0.)
        elif gui.is_pressed(ti.GUI.RIGHT):
            camera.rotate( 1., 0.)
        elif gui.is_pressed(ti.GUI.UP):
            camera.rotate( 0., 1.)
        elif gui.is_pressed(ti.GUI.DOWN):
            camera.rotate( 0.,-1.)
        if gui.is_pressed('w'):
            camera.move(0., 0.01, 0)
        elif gui.is_pressed('a'):
            camera.move(-0.01,0., 0.)
        elif gui.is_pressed('s'):
            camera.move(0.,-0.01, 0)
        elif gui.is_pressed('d'):
            camera.move(0.01, 0., 0.)
        # mouse interaction
        if gui.is_pressed(ti.GUI.LMB):
            self.mouse_pos = gui.get_cursor_pos()
            if self.picked == -1:
                self.picked = pick(pos2, self.mouse_pos)
        else:
            self.picked = -1
