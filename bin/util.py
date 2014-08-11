
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
