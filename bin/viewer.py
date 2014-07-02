
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import SWC

class MPLviewer:

    def __init__(self, masked_img, swc, condition):
        self.fig = plt.figure(1)
        self.img = masked_img
        self.gs = gspec.GridSpec(10, 10)
        self.imgdisplay = self.fig.add_subplot(gs[:9, :])
        if len(masked_img.shape) == 3:
            self.index = 0
            self.max_index = masked_img.shape[0] - 1
            self.imgshown = self.imgdisplay.imshow(masked_img[self.index], 
                                                   interpolation='none')
            self.x_max = self.img[2]
            self.y_max = self.img[1]
        elif len(masked_img.shape) == 2:
            self.index = 0
            self.max_index = 0
            self.imgshown = self.imgdisplay.imshow(masked_img, 
                                                   interpolation='none')
            self.x_max = self.img[1]
            self.y_max = self.img[0]
        else:
            plt.close(self.fig)
            return
        self.segs = self._process_swc(swc)
        self.fig.canvas.mpl_connect('key_press_event', self._key_press)
        self._draw_vessel_cross()
        plt.draw()
        
    def _process_swc(swc): 
        segs = np.zeros((self.max_index), dtype=np.dtype('object'))
        for segment in swc:
            for piece in segment:
                z_coord = piece.z
                occupant = segs[z_coord]
                if occupant == 0:
                    segs[z_coord] = np.array([piece])
                else:
                    segs[z_coord] = np.append(occupant, piece)
        return segs

    def _key_press(self, event):
        if event.key == 'left':
            self._move_view(-1)
        elif event.key == 'right':
            self._move_view(1)
        elif event.key == 'escape':
            plt.close(self.fig)
        self._update()

    def _move_view(self, mod):
        if self.index + mod >= 0 and self.index + mod <= self.max_index:
            self.index = self.index + mod
            self._update()

    def _draw_vessel_cross(self):
        # potentially add color derived from ident
        to_draw = self.segs[self.index]
        for item in to_draw:
            self.imgdisplay.plot([item.y],[item.x], marker=r'&\bigoplus$',
                                 markersize=item.rad*2)
        self.imgdisplay.set_ylim(0, self.y_max)
        self.imgdisplay.set_xlim(0, self.x_max)
        
    def _update(self):
        self.imgshown.set_data(self.img[self.index])
        self._draw_vessel_cross()
        plt.draw()

    
