
from time import time

def timeThis(func):
    def wapper(*args, **kargs):
        s = time()
        result = func(*args, **kargs)
        print(f'{func.__name__!r} executed in {(time()-s):.4f}s')
        return result
    return wapper