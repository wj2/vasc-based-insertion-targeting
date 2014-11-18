
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

def create_probes(sizes, rotations, buffs, mpp=None):
    # sizes here includes depth, in the form (depth, length, width), will be 
    # converted to tuple to make hashable
    # rotations are (xy, yz, xz) tuples
    if mpp is None:
        mpp = np.ones((1, 3)).reshape(1, 3)
    probes = {}
    y_off, x_off = 0,0
    print sizes
    for s in sizes:
        s = tuple(s)
        probes[s] = {}
        for b in buffs:
            probes[s][b] = {}
            for r in rotations:
                xy, yz, xz = r
                probe = create_probe(s[1:], s[0], mpp=mpp, xy=xy, yz=yz, xz=xz)
                probes[s][b][r] = probe
                y_off = max(y_off, int(np.ceil(probe.shape[1] / 2.)))
                x_off = max(x_off, int(np.ceil(probe.shape[2] / 2.)))
    return probes, y_off, x_off

def create_probe(size, depth, xy=0, yz=0, xz=0, buff=0, mpp=None):
    if mpp is None:
        mpp = (1., 1., 1.)
    probe = np.ones(((depth+buff) / mpp[2], (size[0]+buff)/mpp[1], 
                     (size[1]+buff)/mpp[0]+1))
    probe[:, :, -1] = 0
    probe = rotate_probe(probe, xy, yz, xz)
    return probe

def rotate_probe(p, xy, yz, xz):
    p = rotate(p, xz, axes=(2, 0))
    p = rotate(p, yz, axes=(1, 0))
    p = rotate(p, xy, axes=(2,1))
    return np.around(p)

class ProbePlacer(object):
    
    def __init__(self, probesize, probedepth, mpp, stackpath=None, stack=None, 
                 collapse=1, ind=0):
        if stackpath is None and stack is None:
            raise IOError('one of stackpath or stack is required')
        elif stack is None:
            stack = tiff.imread(stackpath)
        self.stack = util.collapse_stack(stack, collapse)
        self.collapse = collapse
        self._max_i = self.stack.shape[0] - 1
        self._ci = ind
        self._maxx = self.stack.shape[2] - 1
        self._maxy = self.stack.shape[1] - 1
        self.probe = np.ones((probedepth / collapse*mpp[2], 
                              np.around(probesize[0]/mpp[1]), 
                              np.around(probesize[1]/mpp[0] + 1)))
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
        self._im_ax = self._fig.add_subplot(1,1,1, xscale='linear', 
                                            yscale='linear')
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

    def _update_view(self):
        self._im.set_data(self.stack[self._ci])
        self._draw_probe()
        self._im_ax.set_xlim(0, self._maxx)
        self._im_ax.set_ylim(self._maxy, 0)
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
                         'offset':self._p_vert_offset}
            plt.ioff()
            plt.close()
        elif event.key in CLOSE_GUI_END:
            plt.ioff()
            plt.close()

    def get_probe(self, xy=None, yz=None, xz=None, buff=0):
        if xy is None:
            xy = self._p_xy_ang
        if yz is None:
            yz = self._p_yz_ang
        if xz is None:
            xz = self._p_xz_ang
        if buff > 0:
            p = create_probe(self.probe.shape[1:], self.probe.shape[0],
                             xy, yz, xz, buff)
        else:
            p = rotate_probe(self.probe, xy, yz, xz)
        return p

    def get_probe_old(self, xy=None, yz=None, xz=None, buff=0):
        if xy is None:
            xy = self._p_xy_ang
        if yz is None:
            yz = self._p_yz_ang
        if xz is None:
            xz = self._p_xz_ang
        if buff > 0:
            self.probe = np.ones((self.probe.shape[0]+buff, 
                                  self.probe.shape[1]+buff,
                                  self.probe.shape[2]+buff+1))
            self.probe = self.probe[:, :, -1] = 0
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
