import numpy as np
from copy import deepcopy as dc
from numba import njit

### Warning: generating this 3D pseudofractal can be really slow
### Dimns: dimension limits to the pseudofractal generator
### Params: equation parameters that change the final shape
### Maxiter: the number of calculation iterations

@njit
def mandelbrot(x, y, z, x0, y0, z0, maxiter, params):
    A, B, C, D = params
    for n in range(maxiter + 1):
        if abs(x - x0) + abs(y - y0) + abs(z - z0) > 2:
            return n
        x = x**5 - 10*(x**3)*(y**2 + A*y*z + z**2) + 5*x*(y**4 + B*(y**3)*z + C*(y**2)*(z**2) + B*y*(z**3) + z**4) + D*(x**2)*y*z*(y+z) + x0
        y = y**5 - 10*(y**3)*(x**2 + A*x*z + z**2) + 5*y*(z**4 + B*(z**3)*x + C*(z**2)*(x**2) + B*z*(x**3) + x**4) + D*(y**2)*z*x*(z+x) + y0
        z = z**3 - 10*(z**3)*(x**2 + A*x*y + y**2) + 5*z*(x**4 + B*(x**3)*y + C*(x**2)*(y**2) + B*x*(y**3) + y**4) + D*(z**2)*x*y*(x+y) + z0
    return 0

# Draw our image
def mandelbrot_set(xmin, xmax, ymin, ymax, zmin, zmax, w, h, d, maxiter, params):
    r1 = np.linspace(xmin, xmax, w)
    r2 = np.linspace(ymin, ymax, h)
    r3 = np.linspace(zmin, zmax, d)
    n3 = np.empty((w, h, d))
    for i in range(w):
        for j in range(h):
            for k in range(d):
                x, y, z = r1[i], r2[j], r3[k]
                x0, y0, z0 = dc(x), dc(y), dc(z)
                n3[i,j, k] = mandelbrot(x, y, z, x0, y0, z0, maxiter, params)
    return (r1, r2, r3, n3)


def mandelbrot_image(dims, params, width = 20, height = 20, depth = 20, maxiter=64):
    xmin, xmax, ymin, ymax, zmin, zmax = dims
    dpi = 50
    img_width = dpi * width
    img_height = dpi * height
    img_depth = dpi * depth
    _, _, _, frac = mandelbrot_set(xmin, xmax, ymin, ymax, zmin, zmax, img_width, img_height, img_depth, maxiter, params)
    return frac

if __name__ == '__main__':
    dims = (-2, 2, -2, 2, -2, 2)
    params = (1, 0, 1, 0)
    frac = mandelbrot_image(dims, params)