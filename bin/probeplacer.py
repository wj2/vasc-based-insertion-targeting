
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy.ndimage import rotate
import tiff.tifffile as tiff

import util
from probeplacer_keymap import *

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
        self.stack = util.collapse_stack(stack, collapse)
        self.collapse = collapse
        self._max_i = self.stack.shape[0] - 1
        self._ci = ind
        self._maxx = self.stack.shape[2] - 1
        self._maxy = self.stack.shape[1] - 1
        self.probe = np.ones((probedepth / collapse, probesize[1], 
                              probesize[0] + 1))
        print self.probe[:, :, -1]
        self.probe[:, :, -1] = 0
        self._rotated_probe = self.probe
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
                                      interpolation='none',
                                      cmap='gray')
        racm = make_red_alpha_scale_cm()
        self._pim = self._im_ax.imshow(self._rotated_probe[self._ci + 
                                                  self._p_vert_offset],
                                       cmap=racm)
        self._fig.canvas.mpl_connect('key_press_event', self._key_press)
        self._fig.canvas.mpl_connect('button_press_event', self._on_press)
        self._fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self._update_view()
        plt.show()

    def _coord_to_extent(self, x, y):
        return [x - (self._rotated_probe.shape[2] / 2),
                x + (self._rotated_probe.shape[2] / 2),
                y - (self._rotated_probe.shape[1] / 2),
                y + (self._rotated_probe.shape[1] / 2)]

    def _draw_probe(self):
        self._pim.set_data(self._rotated_probe[self._ci + self._p_vert_offset])
        self._pim.set_extent(self._coord_to_extent(self._px, self._py))
        self._im_ax.set_xlim(0, self._maxx + 1)
        self._im_ax.set_ylim(self._maxy + 1, 0)

    def _update_view(self):
        self._draw_probe()
        self._im.set_data(self.stack[self._ci])
        plt.draw()

    def _rotate_probe(self, mod, axes):
        if axes == 'xz': # axes (2, 0)
            self._p_xz_ang += mod
        elif axes == 'yz': # axes (1, 0)
            self._p_yz_ang += mod
        elif axes == 'xy': # axes (2, 1)
            self._p_xy_ang += mod
        self._rotated_probe = rotate(self.probe, self._p_xz_ang, axes=(2, 0))
        self._rotated_probe = rotate(self._rotated_probe, self._p_yz_ang, 
                                     axes=(1, 0))
        self._rotated_probe = rotate(self._rotated_probe, self._p_xy_ang, 
                                     axes=(2,1))
        print self._rotated_probe
        print self._rotated_probe.shape
        print self._rotated_probe.max()
        print self._rotated_probe.min()
        self._update_view()

    def _move_probe(self, mod, axis):
        if axis == 'x':
            self._px += mod
        elif axis == 'y':
            self._py += mod
        elif axis == 'z':
            self._p_vert_offset += mod
        self._update_view()

    def _move_view(self, mod):
        result = self._ci + mod
        if result >= 0 and result <= self._max_i:
            self._ci += mod
        self._update_view()

    def _key_press(self, event):
        print event.key
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
        elif event.key in ROTATE_XZ_CCW:
            self._rotate_probe(-1, 'xz')
        elif event.key in ROTATE_XZ_CW:
            self._rotate_probe(1, 'xz')
        elif event.key in ROTATE_YZ_CCW:
            self._rotate_probe(-1, 'yz')
        elif event.key in ROTATE_YZ_CW:
            self._rotate_probe(1, 'yz')
        elif event.key in ROTATE_XY_CCW:
            self._rotate_probe(-1, 'xy')
        elif event.key in ROTATE_XY_CW:
            self._rotate_probe(1, 'xy')
        elif event.key in CLOSE_GUI_FINISHED:
            self.info = {'xy_ang':self._p_xy_ang, 'xz_ang':self._p_xz_ang,
                         'yz_ang':self._p_yz_ang, 'x':self._px, 'y':self._py,
                         'offset':self._vert_offset}
        else:
            print 'missed'

    def get_probe(self, xy=self._p_xy_ang, yz=self._p_yz_ang, 
                  xz=self._p_xz_ang):
        p = rotate(self.probe, xz, axes=(2, 0))
        p = rotate(p, yz, axes=(1, 0))
        p = rotate(p, xy, axes=(2,1))
        return np.around(p)

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
