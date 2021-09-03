import taichi as ti

def vec2(x=0., y=0.):
    return ti.Vector([x,y])

def vec3(x=0., y=0., z=0.):
    return ti.Vector([x,y,z])

def arr2(x=0., y=0.):
    return [x,y]

def arr3(x=0., y=0., z=0.):
    return [x,y,z]

def mat2(v1=arr2(),v2=arr2(), byCol=False):
    ret = ti.Matrix([list(v1), list(v2)])
    if byCol:
        ret = ret.transpose()
    return ret

def mat3(v1=arr3(),v2=arr3(),v3=arr3(), byCol=False):
    ret = ti.Matrix([list(v1), list(v2), list(v3)])
    if byCol:
        ret = ret.transpose()
    return ret

def field(shape=(), dim=3, dtype=ti.f32):
    if isinstance(dim, tuple) and len(dim) == 2:
        return ti.Matrix.field(dim[0], dim[1], dtype=dtype, shape=shape)
    elif isinstance(dim, int): 
        if dim > 1:
            return ti.Vector.field(dim, dtype=dtype, shape=shape)
        else:
            return ti.field(dtype, shape=shape)
    else:
        raise NotImplementedError(f'Dim type {type(dim)} is not supported')
