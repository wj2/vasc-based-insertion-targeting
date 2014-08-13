
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy.ndimage import rotate
import tiff.tifffile as tiff

import util
from prepostmatcher_keymap import *

def make_greenscale_cm():
    reddict = {'red' : [(0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)],
               'green' : [(0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0)],
               'blue' : [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)]}
    greenscale = color.LinearSegmentedColormap('greenscale', reddict)
    return greenscale

def make_redscale_cm():
    reddict = {'red' : [(0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0)],
               'green' : [(0.0, 0.0, 0.0),
                          (1.0, 0.0, 0.0)],
               'blue' : [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)]}
    redscale = color.LinearSegmentedColormap('redscale', reddict)
    return redscale    

class PrePostMatcher(object):

    def __init__(self, pre, post, collapse=1, start_index=0):
        self.pre = util.collapse_stack(tiff.imread(pre), collapse)
        self.post = util.collapse_stack(tiff.imread(post), collapse)
        self._ci = start_index
        self._max_i = self.post.shape[0] - 1
        self._offset = 0
        self._x = self.pre.shape[2] / 2
        self._y = self.pre.shape[1] / 2
        self._fig = plt.figure()
        self._im_ax = self._fig.add_subplot(1,1,1)
        self._selected = False
        gcm = make_greenscale_cm()
        rcm = make_redscale_cm()
        self._postim = self._im_ax.imshow(self.post[self._ci], 
                                          interpolation='none', cmap=rcm, 
                                          alpha=.5)
        ext = self._coord_to_extent(self._x, self._y, self.pre)
        self._pre_ang = 0
        self._pre_rot = self.pre
        self._preim = self._im_ax.imshow(self._pre_rot[self._ci], 
                                         interpolation='none', cmap=gcm, 
                                         alpha=.5, extent=ext)
        self._fig.canvas.mpl_connect('key_press_event', self._key_press)
        self._fig.canvas.mpl_connect('button_press_event', self._on_press)
        self._fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self._update_view()
        plt.show()

    def _coord_to_extent(self, x, y, stack):
        return [x-stack.shape[2] / 2, x+stack.shape[2] / 2,
                y+stack.shape[1] / 2, y-stack.shape[1] / 2]

    def _move_overlay(self, mod, dim):
        if dim == 'x':
            self._x += mod
        elif dim == 'y':
            self._y += mod
        self._update_view()

    def _adjust_offset(self, mod):
        self._offset += mod
        self._update_view()

    def _adjust_depth(self, mod):
        result = self._ci + mod
        if 0 <= result <= self._max_i:
            self._ci += mod
            self._update_view()

    def _rotate_stack(self, mod):
        self._pre_ang += mod
        self._pre_rot = rotate(self.pre, self._pre_ang, axes=(2, 1), 
                               reshape=False)
        self._update_view()

    def _key_press(self, event):
        # need xy rotation, increase/decrease offset, xy translation,
        # up/down in stacks
        if event.key in LEFT:
            self._move_overlay(-1, 'x')
        elif event.key in RIGHT:
            self._move_overlay(1, 'x')
        elif event.key in FORWARD:
            self._move_overlay(-1, 'y')
        elif event.key in BACKWARD:
            self._move_overlay(1, 'y')
        elif event.key in UP_OFFSET:
            self._adjust_offset(-1)
        elif event.key in DOWN_OFFSET:
            self._adjust_offset(1)
        elif event.key in UP_STACK:
            self._adjust_depth(-1)
        elif event.key in DOWN_STACK:
            self._adjust_depth(1)
        elif event.key in ROTATE_XY_CW:
            self._rotate_stack(1)
        elif event.key in ROTATE_XY_CCW:
            self._rotate_stack(-1)
        elif event.key in CLOSE_GUI_FINISHED:
            self._finished = True
            x = self._x - self.post.shape[2] / 2
            y = self._y - self.post.shape[1] / 2
            self.info = {'x':x, 'y':y, 'xy_ang':self._pre_ang, 
                         'offset':self._offset}
            plt.close()
        elif event.key in CLOSE_GUI_END:
            self._finished = False
            plt.close()

    def _on_press(self, event):
        x = event.xdata
        y = event.ydata
        ext = self._coord_to_extent(self._x, self._y, self._pre_rot)
        if ext[0] <= x <= ext[1] and ext[3] <= y <= ext[2]:
            print 'clicked within'
            self._selected = True
            self._clickoffset_x = x - self._x # no jump from movement
            self._clickoffset_y = y - self._y

    def _on_release(self, event):
        self._selected = False

    def _on_move(self, event):
        if self._selected:
            self._x = np.around(event.xdata) - self._clickoffset_x
            self._y = np.around(event.ydata) - self._clickoffset_y
            self._update_view()

    def _update_view(self):
        self._postim.set_data(self.post[self._ci])
        self._preim.set_data(self._pre_rot[self._ci + self._offset])
        self._preim.set_extent(self._coord_to_extent(self._x, self._y, 
                                                     self.pre))
        
        self._im_ax.set_xlim(0, self.post.shape[2])
        self._im_ax.set_ylim(self.post.shape[1], 0)
        plt.draw()
    
