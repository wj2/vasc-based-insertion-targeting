
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import tiff.tifffile as tiff
import swc    

class MPLviewer:

    def __init__(self, masked_img, swc, condition=None, start_i=0,
                 display_ids=True):
        """ 
        condition is either None or a function that returns true if
        a given segment should be included and false otherwise
        """
        masked_img = tiff.imread(masked_img)
        self.fig = plt.figure()
        self.img = masked_img
        self.gs = gspec.GridSpec(10, 10)
        self.imgdisplay = self.fig.add_subplot(self.gs[:, :])
        self.display_ids = display_ids
        if len(masked_img.shape) == 3:
            self.index = start_i
            self.max_index = masked_img.shape[0] - 1
            self.imgshown = self.imgdisplay.imshow(masked_img[self.index], 
                                                   interpolation='none')
            self.x_max = self.img.shape[2]
            self.y_max = self.img.shape[1]
        elif len(masked_img.shape) == 2:
            self.index = 0
            self.max_index = 0
            self.imgshown = self.imgdisplay.imshow(masked_img, 
                                                   interpolation='none')
            self.x_max = self.img.shape[1]
            self.y_max = self.img.shape[0]
        else:
            plt.close(self.fig)
            return
        self.segs = self._process_swc(swc, condition)
        self.fig.canvas.mpl_connect('key_press_event', self._key_press)
        self._draw_vessel_cross()
        self.imgdisplay.set_title(str(self.index) + '/' + str(self.max_index))
        plt.draw()
        
    def _process_swc(self, swc, condition): 
        if condition is not None:
            swc = swc.filter(condition)
        segs = np.zeros((self.max_index + 1), dtype=np.dtype('object'))
        for segment in swc:
            for piece in segment:
                z_coord = piece.z
                if z_coord <= self.max_index:
                    occupant = segs[z_coord]
                    if np.all(occupant == 0):
                        segs[z_coord] = np.array([piece])
                    else:
                        segs[z_coord] = np.append(occupant, piece)
        return segs

    def rads_by_layer(self):
        layers = []
        for layer in self.segs:
            if np.all(layer != 0):
                rs = [item.rad for item in layer]
                layers.append(np.median(rs))
            else:
                layers.append(-1)
        return layers

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
        mark = r'$\bigoplus$'
        to_draw = self.segs[self.index]
        [x.remove() for x in self.imgdisplay.lines]
        if np.all(to_draw != 0):
            for item in to_draw:
                if self.display_ids:
                    mark = '$'+str(item.seg)+'$'
                    msize = 35
                else:
                    msize = item.rad * 2
                self.imgdisplay.plot([item.x],[item.y], 'g',
                                     marker=mark,
                                     markersize=msize)
        self.imgdisplay.set_ylim(self.y_max, 0)
        self.imgdisplay.set_xlim(0, self.x_max)
        
    def _update(self):
        self.imgshown.set_data(self.img[self.index])
        self._draw_vessel_cross()
        self.imgdisplay.set_title(str(self.index) + '/' + str(self.max_index))
        plt.draw()

   
