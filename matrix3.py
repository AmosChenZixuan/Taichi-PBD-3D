import taichi as ti


def vec3(x=0., y=0., z=0.):
    return ti.Vector([x,y,z])

def mat3():
    return ti.Vector([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]])

@ti.func
def getCol0(mat):
    ret = vec3()
    ret[0] = mat[0, 0]
    ret[1] = mat[1, 0]
    ret[2] = mat[2, 0]
    return ret

@ti.func
def getCol1(mat):
    ret = vec3()
    ret[0] = mat[0, 1]
    ret[1] = mat[1, 1]
    ret[2] = mat[2, 1]
    return ret

@ti.func
def getCol2(mat):
    ret = vec3()
    ret[0] = mat[0, 2]
    ret[1] = mat[1, 2]
    ret[2] = mat[2, 2]
    return ret

@ti.pyfunc
def setCol0(mat, trg):
    mat[0, 0] = trg[0]
    mat[1, 0] = trg[1]
    mat[2, 0] = trg[2]
    return mat

@ti.pyfunc
def setCol1(mat, trg):
    mat[0, 1] = trg[0]
    mat[1, 1] = trg[1]
    mat[2, 1] = trg[2]
    return mat

@ti.pyfunc
def setCol2(mat, trg):
    mat[0, 2] = trg[0]
    mat[1, 2] = trg[1]
    mat[2, 2] = trg[2]
    return mat