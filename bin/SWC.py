
import numpy as np
import matplotlib.pyplot as plt
from os.path import splitext
from compare import angle_between

"""
Classes: SWC, SuperSegment, Segment, Piece, Vertex

SuperSegment contains Segment(s) and Vertex(s)

SWC inherits from SuperSegment and reads .swc or .eswc files

Segment contains Piece(s)


"""

def create_super_seg(indices, swc, ident=None):
    return SuperSegment([swc[i] for i in indices], ident=ident)

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
    while child != False:
        child.seg = parentest.seg
        segment.add(child)
        child = all_.get(child.ident, False)
    return segment

def _delve_to_children_rec(all_, parentest, segment):
    child = all_.get(parentest.ident, False)
    if child == False:
        return segment
    else:
        child.seg = parentest.seg
        segment.add(child)
        return _delve_to_children(all_, child, segment)

def dist(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    return np.sqrt(np.sum((c1 - c2) ** 2))


class SuperSegment(object):
    
    def __init__(self, segs, ident=None):
        self._vertices = []
        self.segs = segs
        self.num_segs = len(segs)
        if ident is None:
            self.ident = [x.ident for x in args]
        else:
            self.ident = ident

    def length(self):
        return np.sum([x.length() for x in self.segs])

    def avg_radius(self):
        return np.average([x.avg_radius() for x in self.segs],
                          weights=[x.length() for x in self.segs])

    def downwardness(self, crow=True):
        return np.average([x.downwardness(crow) for x in self.segs],
                          weights=[x.length() for x in self.segs])

    def inspect(self, print_=True):
        down_c = self.downwardness()
        down_w = self.downwardness(False)
        rad_c = self.avg_radius()
        length_c = self.length()
        if print_:
            print 'seg id: ' + str(self.ident)
            print 'down c: ' + str(down_c)
            print 'down w: ' + str(down_w)
            print 'rad   : ' + str(rad_c)
            print 'len   : ' + str(length_c)
        
        return [self.ident, down_c, down_w, rad_c, length_c]

    def array_rep(self):
        coords = self.xyzs
        dims = np.max(coords, axis=0)[::-1] + 1
        seg_arr = np.zeros(dims) # reverse for zyx
        pie_arr = np.zeros(dims)
        for segment in self:
            seg_arr, pie_arr = segment.array_rep(seg_arr, pie_arr)
        return seg_arr, pie_arr

    def _find_edges(self, x, y, z, seg, pie, edges, radius=5): 
        already_segs = set([x[0] for x in edges])
        seg_sect = seg[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius]
        pie_sect = pie[z-radius:z+radius, y-radius:y+radius, x-radius:x+radius]
        vess = set(seg_sect[seg_sect > 0]).difference(already_segs)
        for v in vess:
            # find piece closest to x,y,z
            cs = np.where(seg_sect == v)
            dists = map(lambda x: dist((x, y, z), x), zip(*cs))
            c = argmin(dists)
            pid = pie_sect[c[0], c[1], c[2]]
            edges.append((v, pid))
        return edges

    def vertexify(self, seg_arr=None, pie_arr=None):
        if seg_arr is None and pie_arr is None:
             seg_arr, pie_arr = self.array_rep()
        for segment in self:
            start = segment[0]
            old_es = []
            es = [(start.seg, start.ident)]
            while len(es) > len(old_es):
                old_es = es
                es = self._find_edges(start.x, start.y, start.z, seg_arr, 
                                      pie_arr, es)
            self.vertices.append(Vertex.from_edges(es, self))
            end = segment[-1]
            old_es = []
            es = [(end.seg, end.ident)]
            while len(es) > len(old_es):
                old_es = es
                es = self._find_edges(end.x, end.y, end.z, seg_arr, 
                                      pie_arr, es)
            self.vertices.append(Vertex.from_edges(es, self))
            
            
    @property
    def vertices(self):
        if len(self._vertices) == 0:
            raise Exception('vertices have not been computed')
        return self._vertices

    @property
    def root(self):
        return self.__getitem__(0)[0]

    @property
    def rads(self):
        return np.concatenate([x.rads for x in self.segs])
    
    @property
    def micron_rads(self):
        return np.concatenate([x.micron_rads for x in self.segs])

    @property
    def xyzs(self):
        return np.concatenate([x.xyzs for x in self.segs])

    @property
    def xs(self):
        return np.concatenate([x.xs for x in self.segs])

    @property
    def ys(self):
        return np.concatenate([x.ys for x in self.segs])

    @property
    def zs(self):
        return np.concatenate([x.zs for x in self.segs])


class SWC(SuperSegment):
    
    def __init__(self, path=None, segments=None, microns_perpixel=1, 
                 cylinder=True, ident=None):
        """ must have one of segments or path or both """
        self.micsperpix = microns_perpixel
        if path is None and segments is not None:
            super(SWC, self).__init__(segments, ident)
        elif path is not None:
            self._define_by_path(path, cylinder, segments, ident)
        
    def _define_by_segments(self, segments):
        self.segs = segments
        self.num_segs = len(segments) 
        
    def _define_by_path(self, path, cylinder, segs, ident):
        self._vertices = []
        if ident is None:
            self.ident = path
        if segs is None:
            self.segs = []
            self.num_segs = 0
        else:
            self.segs = segs
            self.num_segs = len(segs)
        dict_all = {}
        with open(path, 'rb') as swc_text:
            for entry in swc_text:
                if entry[0] == '#':
                    pass
                else:
                    swcent = Piece.from_string(entry, cylinder)
                    if swcent.par == -1:
                        s = Segment(self.micsperpix, pieces=np.array([swcent]))
                        self.segs.append(s)
                        self.num_segs += 1
                    else:
                        dict_all[swcent.par] = swcent

        # now we have all parents in one dict and all other segments in another
        # indexed by parent
        for i, value in enumerate(self.segs):
            value.ident = i + 1 # no zero seg value
            value[0].seg = i
            most_parentest = value[0]
            self.segs[i] = _delve_to_children(dict_all, most_parentest, 
                                              value)
            
    def __getitem__(self, key):
        return self.segs[key]

    def __len__(self):
        return self.num_segs

    def __iter__(self):
        return iter(self.segs)

    def filter(self, function):
        new_segs = filter(function, self)
        return SWC(segments=new_segs, microns_perpixel=self.micsperpix)
        
class Segment(object): 
    
    def __init__(self, micsperpix, ident=-1, pieces=None):
        self.mpp = micsperpix
        self.ident = ident
        if pieces is None:
            self._pieces = np.array([])
            self._num_pieces = 0
        else:
            self._pieces = np.array(pieces)
            self._num_pieces = len(pieces)
        
    def __getitem__(self, key):
        return self._pieces[key]

    def __iter__(self):
        return iter(self._pieces)
    
    def __len__(self):
        return self._num_pieces

    def inspect(self, print_=True):
        down_c = self.downwardness()
        down_w = self.downwardness(False)
        rad_c = self.avg_radius()
        length_c = self.length()
        if print_:
            print 'seg id: ' + str(self.ident)
            print 'down c: ' + str(down_c)
            print 'down w: ' + str(down_w)
            print 'rad   : ' + str(rad_c)
            print 'len   : ' + str(length_c)
        
        return [self.ident, down_c, down_w, rad_c, length_c]

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
        if attr is None:
            plt.plot(self.micron_rads)
            plt.xlabel('segment number')
        else:
            plt.plot(getattr(self, attr), self.micron_rads)
            plt.xlabel(attr)
        plt.ylabel('radius')
        plt.show()

    def downwardness(self, crow=True):
        """ 
        returns difference in radians between direction of segment and 
        (0, 0, 1) -- expect values between [0, pi/2]; pi/2 = 1.570796
          - direction can be either as crow flies (default) or weighted average
        """
        if crow:
            avg_diff = self.crow_direction()
        else: 
            avg_diff = self.avg_direction()
        if avg_diff[-1] < 0:
            avg_diff = avg_diff * -1
        return angle_between(avg_diff, np.array([0, 0, 1]))

    def _piece_lens(self, diffs):        
        return np.sqrt(np.sum(diffs**2, axis=1))

    def _xyz_diffs(self):
        xyzs = self.xyzs
        return np.diff(xyzs, axis=0)

    def avg_radius(self, weighted=False):
        if weighted:
            dxyzs = self._xyz_diffs()
            lxyzs = self._piece_lens(dxyzs)
            avgrad = np.average(self.rads[1:], axis=0, weights=lxyzs)
        else:
            avgrad = np.mean(self.rads)
        return avgrad * self.mpp
            
    def avg_direction(self):
        """ length weighted average direction of segment """
        diff_xyzs = self._xyz_diffs()
        length_xyzs = self._piece_lens(diff_xyzs)
        return np.average(diff_xyzs, axis=0, weights=length_xyzs)

    def cumu_direction(self):
        diff_xyzs = self._xyz_diffs()
        length_xyzs = self._piece_lens(diff_xyzs)
        return

    def length(self):
        """ length of segment, calculated going point to point """
        diff_xyzs = self._xyz_diffs()
        return self._piece_lens(diff_xyzs).sum() * self.mpp

    def crow_length(self):
        """ length of segment as crow flies, from first to last piece """
        return np.sqrt(np.sum((np.array(self[0].xyz) 
                               - np.array(self[-1].xyz))**2)) * self.mpp

    def crow_direction(self):
        """ directionality of segment, taken from first to last piece only """
        dir_ = np.array(self[-1].xyz) - np.array(self[0].xyz)
        if dir_[-1] < 0:
            dir_ = dir_ * -1
        return dir_

    def split(self, piece_id=None, piece_num=None):
        if piece_id is None and piece_num is not None:
            i_of = piece_num
        elif seg_id is not None and piece_num is None:
            i_of = map(lambda x: x.ident, self._pieces).index(piece_id)
        else:
            raise Exception('split takes either piece_id or piece_num, not '
                            'both or neither')
        new_seg = Segment(self.mpp, pieces=self._pieces[i_of:])
        self._pieces = self._pieces[:i_of+1]
        self._num_pieces = len(self._pieces)
        return new_seg
    
    def array_rep(self, seg_arr=None, pie_arr=None):
        if seg_arr is None and pie_arr is None:
            coords = self.xyzs
            dims = coords.max(axis=0)[::-1]
            seg_arr = np.zeros(dims)
            pie_arr = np.zeros(dims)
        for piece in self:
            xyz = piece.xyz
            r_zyx = np.around(xyz)[::-1]
            seg_arr[r_zyx[0], r_zyx[1], r_zyx[2]] = self.ident
            pie_arr[r_zyx[0], r_zyx[1], r_zyx[2]] = piece.ident
        return seg_arr, pie_arr
            
    @property
    def root(self):
        return self.__getitem__(0)

    @property
    def rads(self):
        return self._get_attr_list('rad')

    @property
    def micron_rads(self):
        return self.rads * self.mpp

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


class Piece(object):
    
    def __init__(self, attributes, cylinder):
        ident, struct, x, y, z, rad, par = attributes[:7]
        self.cylinder = cylinder
        self.ident = int(ident)
        self.struct = int(struct)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.rad = float(rad)
        self.par = int(par)
        if len(attributes) > 7:
            self.seg = attributes[7]
        else:
            self.seg = -1

    @classmethod
    def from_string(cls, entry, cylinder):
        return cls(entry.split(' '), cylinder)

    def get_dimmin(self, dim):
        if self.cylinder:
            dimmin = getattr(self, dim) + 0.5
        else:
            dimmin = getattr(self, dim) - self.rad
        return dimmin

    def get_dimmax(self, dim):
        if self.cylinder:
            dimmax = getattr(self, dim) + 0.5
        else:
            dimmax = getattr(self, dim) + self.rad
        return dimmax

    @property
    def xyz(self):
        return (self.x, self.y, self.z)

    @property
    def is_vertex(self):
        return False

    @property
    def attributes(self):
        return self.ident, self.struct, self.x, self.y, self.z, self.rad, \
            self.par, self.seg


class Vertex(object):

    def __init__(self, x, y, z, r, edges):
        self.x = x
        self.y = y
        self.z = z
        self.rad = r
        self.edges = edges
        self.degree = len(edges)

    @classmethod
    def from_piece(cls, piece):
        x, y, z = piece.xyz
        r = piece.rad
        es = [(piece.seg, piece.ident)]
        return cls(x, y, z, r, es)

    @classmethod
    def from_edges(cls, edges, swc):
        e1 = swc[edges[0][0]][edges[0][1]]
        v = cls(e1.x, e1.y, e1.z, e1.rad, edges[0])
        return v.add_edges(edges[1:], swc)

    def add_edge_bypiece(self, piece):
        self.add_edge(piece.x, piece.y, piece.z, piece.rad, 
                      (piece.seg, piece.ident))

    def add_edge(self, x, y, z, r, e):
        self.x = (self.x + x) / 2.
        self.y = (self.y + y) / 2.
        self.z = (self.z + z) / 2.
        self.rad = max(self.rad, r)
        self.edges.append(e)
        self.degree = self.degree + 1

    def add_edges(self, es, swc):
        for e in es:
            p = swc[e[0]].piece_id(e[1])
            self.add_edge(p.x, p.y, p.z, p.rad, e))

    @property
    def xyz(self):
        return (self.x, self.y, self.z)
