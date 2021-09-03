import taichi as ti

from include.data import vec2, vec3

@ti.func
def getCol(mat, idx):
    ret = vec3()
    for i in ti.static(range(3)):
        ret[i] = mat[i, idx_by_value__]
    return ret

@ti.func
def setCol(mat, idx, vec):
    assert mat.n == len(vec)
    for i in ti.static(range(3)):
        mat[i, idx_by_value__] = vec[i]
    return mat

@ti.func
def getCol2(mat, idx):
    ret = vec2()
    for i in ti.static(range(2)):
        ret[i] = mat[i, idx_by_value__]
    return ret


if __name__ == '__main__':
    ti.init()

    M = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(1,1))

    M[0,0] = ti.Matrix([
                [0,1,2],
                [3,4,5],
                [6,7,8]
            ])


    @ti.kernel
    def gg():
        print(getCol(M[0,0], 1))
        print(setCol(M[0,0], 2, [8,5,2]))
    gg()
