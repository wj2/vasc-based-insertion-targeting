
import numpy as np

def collapse_stack(stack, collapse):
    if collapse > 1: 
        z_dim, y_dim, x_dim = stack.shape
        z_dim = z_dim / collapse
        new_container = np.zeros((z_dim, y_dim, x_dim))
        for i in xrange(z_dim):
            x = i * collapse
            new_container[i] = np.mean(stack[x:x+z_dim], axis=0)
    else:
        new_container = stack
    return new_container

def rotate_in_plane(xy1, xyc, ang, degrees=True):
    if degrees:
        ang = ang *  np.pi / 180
    xy1 = np.array(xy1)
    xyc = np.array(xyc)
    x, y = xy1 - xyc
    x2 = x * np.cos(ang) - y * np.sin(ang)
    y2 = x * np.sin(ang) + y * np.cos(ang)
    return x2, y2
