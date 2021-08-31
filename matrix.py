import taichi as ti

def vec3(x=0., y=0., z=0.):
    return ti.Vector([x,y,z])

@ti.pyfunc
def getCol(mat, coli):
    rows = ti.static(range(mat.n))
    ret = ti.Vector([0. for _ in rows])
    for i in rows:
        ret[i] = mat[i,coli]
    return ret

@ti.pyfunc
def getRow(mat, rowi):
    cols = ti.static(range(mat.m))
    ret = ti.Vector([0. for _ in cols])
    for i in cols:
        ret[i] = mat[rowi,i]
    return ret

@ti.pyfunc
def setCol(mat, coli, col):
    assert mat.n == len(col)
    for i in ti.static(range(mat.n)):
        mat[i,coli] = col[i]
    return mat

@ti.pyfunc
def setRow(mat, rowi, row):
    assert mat.m == len(row)
    for i in ti.static(range(mat.m)):
        mat[rowi,i] = row[i]
    return mat


if __name__ == '__main__':
    ti.init()

    M = ti.Matrix.field(4, 3, dtype=ti.f32, shape=1)

    setCol(M[0], 0, ti.Vector([1.,2.,3.,-1]))
    M[0] = setCol(M[0], 1, ti.Vector([4.,5.,6.,-2]))
    setCol(M[0], 2, ti.Vector([7.,8.,9.,-3]))

    print(M[0].value)
    print(getCol(M[0], 1))
    print(getRow(M[0], 2))

    setRow(M[0], 3, ti.Vector([15.,20.,25.]))
    print(getRow(M[0], 3))
    print(M[0].value)