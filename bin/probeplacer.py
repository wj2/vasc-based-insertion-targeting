
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import tifffile.tiff as tiff

from manipkeys import *

def make_red_alpha_scale_cm():
    redalphadict = {'red' : [(0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0)],
               'green' : [(0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0)],
               'blue' : [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)],
               'alpha' : [(0.0, 0.0, 0.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 1.0, 1.0)]}
    redalphascale = color.LinearSegmentedColormap('redalphascale', 
                                                  redalphadict)
    return redalphascale    

class ProbePlacer(object):
    
    def __init__(self, stackpath, probesize, probedepth, collapse=1, ind=0):
        stack = tiff.imread(stackpath)
        self.stack = self._collapse_stack(stack, collapse)
        self.collapse = collapse
        self._max_i = self.stack.shape[0] - 1
        self._ci = ind
        self._maxx = self.stack.shape[2] - 1
        self._maxy = self.stack.shape[1] - 1

        self.probe = np.ones((probedepth, probesize[1], probesize[0]))
        self._p_xy_ang = 0
        self._p_xz_ang = 0
        self._p_yz_ang = 0
        self._p_vert_offset = 0
        self._px = self._maxx / 2
        self._py = self._maxy / 2
        self._selected_probe = False
        self._fig = plt.figure()
        self._im_ax = self._fig.add_subplot(1,1,1)
        self._im = self._im_ax.imshow(self.stack[self._ci], 
                                      interpolation='none'
                                      cm='gray')
        racm = make_red_alpha_scale_cm()
        self._pim = self._im_ax.imshow(self.probe[self._ci + 
                                                  self._p_vert_offset],
                                       cm=racm)
        self._fig.canvas.mpl_connect('key_press_event', self._key_press)
        self._fig.canvas.mpl_connect('button_press_event', self._on_press)
        self._fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self._update_view()

    def _collapse_stack(self, stack, collapse):
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

    def _coord_to_extent(self, x, y):
        return [x - (p.shape[2] / 2),
                x + (p.shape[2] / 2),
                y - (p.shape[1] / 2),
                y + (p.shape[1] / 2)]

    def _draw_probe(self):
        self._pim.set_data(self.probe[self._ci + self._p_vert_offset])
        self._pim.set_extent(self._coord_to_extent(self._px, self._py))
        self._im.set_xlim(0, self._maxx + 1)
        self._im.set_ylim(self._maxy + 1, 0)

    def _update_view():
        self._im.set_data(self.stack[self._ci])
        self._draw_probe()
        plt.draw()

    def _rotate_probe(mod, axes):
        # not sure if want to use this code
        # if axes == 'xy':
        #     axes = (1, 2)
        #     result = self._p_xy_ang + mod
        # elif axes == 'yz':
        #     axes = (1,0)
        #     result = self._p_yz_ang + mod
        # elif axes == 'xz':
        #     axes = (2,0)
        #     result = self._p_xz_ang + mod
        # if result in (0, 180):
        #     self.probe = np.ones(())
        self.probe = rotate(self.probe, mod, axes=axes)
        self._update_view()

    def _move_probe(mod, axis):
        if axis == 'x':
            self._px += mod
        elif axis == 'y':
            self._py += mod
        elif axis == 'z':
            self._p_vert_offset += mod
        self._update_view()

    def _move_view(mod):
        result = self._ci + mod
        if result >= 0 and result <= self._max_index:
            self._ci += mod
        self.update_view()

    def _key_press(self, event):
        if event.key in LEFT:
            self._move_probe(-1, 'x')
        elif event.key in RIGHT:
            self._move_probe(1, 'x')
        elif event.key in FORWARD:
            self._move_probe(-1, 'y')
        elif event.key in BACKWARD:
            self._move_probe(1, 'y')
        elif event.key in PROBE_UP:
            self._move_probe(1, 'z')
        elif event.key in PROBE_DOWN:
            self._move_probe(-1, 'z')
        elif event.key in VIEW_UP:
            self._move_view(-1)
        elif event.key in VIEW_DOWN:
            self._move_view(1)
        elif event.key in VERT_XZ_CCW:
            self._rotate_probe(-1, (2, 0))
        elif event.key in VERT_XZ_CW:
            self._rotate_probe(1, (2, 0))
        elif event.key in VERT_YZ_CCW:
            self._rotate_probe(-1, (1, 0))
        elif event.key in VERT_YZ_CW:
            self._rotate_probe(1, (1, 0))
        elif event.key in HORIZ_XY_CCW:
            self._rotate_probe(-1, (2, 1))
        elif event.key in HORIZ_XY_CW:
            self._rotate_probe(1, (2, 1))

    def _on_press(self, event):
        x = event.xdata
        y = event.ydata
        p_extent = self._coord_to_extent(self._px, self._py)
        if (x <= p_extent[1] and x >= p_extent[0] and y <= p_extent[3] 
            and y >= p_extent[2]):
            self._selected_probe = True

    def _on_release(self, event):
        self._selected_probe = False

    def _on_move(self, event):
        if self._selected_probe:
            self._px = np.around(event.xdata)
            self._py = np.around(event.ydata)
            self._update_view()
