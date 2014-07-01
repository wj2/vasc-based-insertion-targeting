
import numpy as np
import matplotlib.pyplot as plt
from os.path import splitext
from SWCEntry import SWCEntry

def add_to_radius(path, add, rad_ind=5):
    with open(path, 'rb') as swcadd:
        pre, suff = splitext(path)
        output = pre + '-radius-add-' + str(add) + suff 
        with open(output, 'wb') as out:
            for line in swcadd:
                line = line.strip()
                if line[0] == '#':
                    pass
                else:
                    line_split = line.split()
                    rad = float(line_split[rad_ind])
                    rad = rad + add
                    line_split[rad_ind] = str(rad)
                    line = ' '.join(line_split)
                out.write(line+'\n')
    return

def _delve_to_children(all_, parentest, segment):
    child = all_.get(parentest.ident, False)
    if child == False:
        return segment
    else:
        segment.add(child)
        return _delve_to_children(all_, child, segment)

def _unit_vector(a):
    return a / np.linalg.norm(a)

def _angle_between(a, b):
    a = _unit_vector(a)
    b = _unit_vector(b)
    return  np.arccos(np.clip(np.dot(a, b), -1, 1))

class SWC:
    
    def __init__(self, path, cylinder=True):
        self._segments = []
        self._num_segments = 0
        dict_all = {}
        with open(path, 'rb') as swc_text:
            for entry in swc_text:
                if entry[0] == '#':
                    pass
                else:
                    swcent = SWCEntry(entry, cylinder)
                    if swcent.par == -1:
                        self._segments.append(Segment(np.array([swcent])))
                        self._num_segments += 1
                    else:
                        dict_all[swcent.par] = swcent

        # now we have all parents in one dict and all other segments in another
        # indexed by parent
        for i, value in enumerate(self._segments):
            most_parentest = value[0]
            self._segments[i] = _delve_to_children(dict_all, most_parentest, 
                                                  value)
            
    def __getitem__(self, key):
        return self._segments[key]

    def __len__(self):
        return self._num_segments

    def __iter__(self):
        return iter(self._segments)
        
class Segment: 
    
    def __init__(self, pieces=None):
        if pieces == None:
            self._pieces = np.array([])
            self._num_pieces = 0
        else:
            self._pieces = np.array(pieces)
            self._num_pieces = 1
        
    def __getitem__(self, key):
        return self._pieces[key]

    def __iter__(self):
        return iter(self._pieces)
    
    def __len__(self):
        return self._num_pieces

    def add(self, piece):
        self._pieces = np.append(self._pieces, piece)
        self._num_pieces += 1
        return

    def _get_attr_list_vectorize(self, attr):
        attrget = np.vectorize(lambda x: getattr(x, attr))
        return attrget(self._pieces)

    def _get_attr_list(self, attr):
        return np.array(map(lambda x: getattr(x, attr), self._pieces))

    def plot_rads_by(self, attr=None):
        if attr == None:
            plt.plot(self.rads)
            plt.xlabel('segment number')
        else:
            plt.plot(getattr(self, attr), self.rads)
            plt.xlabel(attr)
        plt.ylabel('radius')
        plt.show()

    def downwardness(self):
        """ 
        returns difference in radians between mean direction of segment and 
        (0, 0, 1) -- expect values between [0, pi/2]; pi/2 = 1.570796
        """
        avg_diff = self.avg_direction()
        if avg_diff[-1] < 0:
            avg_diff = avg_diff * -1
        return _angle_between(avg_diff, np.array([0, 0, 1]))

    def _piece_lens(self, diffs):        
        return np.sqrt(np.sum(diffs**2, axis=1))

    def _xyz_diffs(self):
        xyzs = self.xyzs
        return np.diff(xyzs, axis=0)

    def avg_radius(self, weighted=False):
        if weighted:
            dxyzs = self._xyz_diffs()
            lxyzs = self._piece_lens(dxyzs)[1:]
            avgrad = np.average(self.rads, axis=0, weights=lxyzs)
        else:
            avgrad = np.mean(self.rads)
        return avgrad
            

    def avg_direction(self):
        """ length weighted average direction of segment """
        diff_xyzs = self._xyz_diffs()
        length_xyzs = self._piece_lens(diff_xyzs)
        return np.average(diff_xyzs, axis=0, weights=length_xyzs)

    def length(self):
        """ length of segment, calculated going point to point """
        diff_xyzs = self._xyz_diffs()
        return reduce(lambda x, y: np.sum(x) + np.sqrt(np.sum(y**2)), 
                      diff_xyzs)

    def crow_length(self):
        """ length of segment as crow flies, from first to last piece """
        return np.sqrt(np.sum((np.array(self[0].xyz) 
                               - np.array(self[-1].xyz))**2))

    def crow_direction(self):
        """ directionality of segment, taken from first to last piece only """
        dir_ = np.array(self[-1].xyz) - np.array(self[0].xyz)
        if dir_[-1] < 0:
            dir_ = dir_ * -1
        return dir_
    
    @property
    def root(self):
        return self.__getitem__(0)

    @property
    def rads(self):
        return self._get_attr_list('rad')

    @property
    def xyzs(self):
        return self._get_attr_list('xyz')

    @property
    def xs(self):
        return self._get_attr_list('x')

    @property
    def ys(self):
        return self._get_attr_list('y')

    @property
    def zs(self):
        return self._get_attr_list('z')
