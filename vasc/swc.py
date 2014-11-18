
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from os.path import splitext
from compare import angle_between
import tiff.tifffile as tiff
from util import memoize
import warnings

"""
Classes: SWC, SuperSegment, Segment, Piece, Vertex

SuperSegment contains Segment(s) and Vertex(s)

SWC inherits from SuperSegment and reads .swc or .eswc files

Segment contains Piece(s)


"""

SEGS_ORDERED_AND_COMPLETE = False

def count_unique(path):
    with open(path, 'rb') as s:
        idents = []
        for line in s:
            if line[0] == '#':
                pass
            else:
                spline = line.split(' ')
                idents.append(int(spline[7]))
    return np.unique(idents)

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
        child.seg = segment.ident
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
        self._vertices = {}
        self.segs = segs
        self.num_segs = len(segs)
        if ident is None:
            self.ident = 'unknown'
        else:
            self.ident = ident

    def __repr__(self):
        rep = '# '+str(self.ident)+'\n'
        rep += ('##n, type, x,y,z,radius,parent,segment_id,segment_layer,'+
                'feature_value \n')
        for seg in self.segs:
            rep += str(seg)
        return rep

    def write(self, filename, eswc=True):
        with open(filename, 'wb') as f:
            f.write(str(self))
        return None

    @memoize
    def segment_id(self, sid):
        for s in self.segs:
            if s.ident == sid:
                return s
        raise Exception('the segment ID '+str(sid)+' is not in this object')

    def add_segment(self, seg):
        self.segs.append(seg)
        self.num_segs += 1
    
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
        seg_arr = np.zeros(dims, dtype=list) # reverse for zyx
        pie_arr = np.zeros(dims, dtype=list)
        for segment in self:
            seg_arr, pie_arr = segment.array_rep(seg_arr, pie_arr)
        return seg_arr, pie_arr

    def _find_edges(self, x, y, z, seg, pie, edges, radius=5, from_end=20, 
                    loop_len=10): 
        already_segs = set([e[0] for e in edges])
        assert len(already_segs) == len(edges)
        assert loop_len < from_end
        x, y, z = np.around(x), np.around(y), np.around(z)
        seg_sect = seg[z-radius:z+radius+1, y-radius:y+radius+1, 
                       x-radius:x+radius+1]
        pie_sect = pie[z-radius:z+radius+1, y-radius:y+radius+1, 
                       x-radius:x+radius+1]
        vs = list(itertools.chain.from_iterable(seg_sect[seg_sect != 0]))
        ps = list(itertools.chain.from_iterable(pie_sect[pie_sect != 0]))
        vess = set(vs)
        vess = vess.difference(already_segs)
        new_edges = []
        for v in vess:
            # find piece closest to x,y,z
            cs = seg_sect.nonzero()
            z_cs = zip(*cs)
            z_cs = [c for c in z_cs if v in seg_sect[c[0],c[1],c[2]]]
            dists = map(lambda c: dist((x, y, z), c), z_cs)
            c = z_cs[np.argmin(dists)]
            ind = seg_sect[c[0], c[1], c[2]].index(v)
            pid = pie_sect[c[0], c[1], c[2]][ind]
            v_seg = self.segment_id(v)
            if (v_seg[0].ident in ps and v_seg[-1].ident in ps and 
                v_seg[0].vertex is None and v_seg[-1].vertex is None):
                if len(v_seg) < loop_len:
                    v_seg[0].vertex = -1
                    v_seg[-1].vertex = -1
                else:
                    new_edges.extend([(v, v_seg[0].ident), 
                                      (v, v_seg[-1].ident)])
            elif pid == v_seg[0].ident and v_seg[0].vertex is None:
                new_edges.append((v, pid))
            elif pid == v_seg[-1].ident and v_seg[-1].ident is None:
                new_edges.append((v, pid))
            elif v_seg[0].ident in ps and v_seg[0].vertex is None:
                new_edges.append((v, v_seg[0].ident))
            elif v_seg[-1].ident in ps and v_seg[-1].vertex is None:
                new_edges.append((v, v_seg[-1].ident))                
            elif (pid not in map(lambda x: x.ident, v_seg[:from_end]) and 
                  pid not in map(lambda x: x.ident, v_seg[-from_end-1:])):
                new_v = v_seg.split(p_id=pid)
                if new_v[-1].vertex is not None:
                    try:
                        up_v = self._vertices[new_v[-1].vertex]
                    except KeyError:
                        print 'keyerror'
                        # if (v_seg.ident, new_v[-1].ident) in edges:
                        e_i = edges.index((v_seg.ident, new_v[-1].ident))
                        edges[e_i] = (new_v.ident, new_v[-1].ident)
                    else:
                        up_v.update_edge((new_v.ident, new_v[-1].ident), 
                                         (v_seg.ident, new_v[-1].ident))
                self.add_segment(new_v)
                new_edges.extend([(v_seg.ident, v_seg[-1].ident), 
                                  (new_v.ident, new_v[0].ident)])
                # we've also got to update the segment map
                nvcs = np.around(new_v.xyzs).astype(int)
                old_ls = seg[nvcs[:, 2], nvcs[:, 1], nvcs[:, 0]]
                old_ps = pie[nvcs[:, 2], nvcs[:, 1], nvcs[:, 0]]
                for i, l in enumerate(old_ls):
                    # ISSUE: segment loops back on itself, this confuses
                    # the indexing
                    ind_p = old_ps[i].index(new_v[i].ident)
                    l[ind_p] = new_v.ident
                # first = np.around(new_v[0].xyz)
                # seg[first[2], first[1], first[0]].append(new_v.ident) 
                # pie[first[2], first[1], first[0]].append(new_v[0].ident)
                # this creates duplication that confuses the above comment
                # issue
                
        return new_edges, seg, pie

    def _seed_edge_search(self, e, seg, pie, radius=5):
        v = Vertex(e.x, e.y, e.z, e.rad, [(e.seg, e.ident)])
        # SEG/IDENT not reflect splittings
        e.vertex = v.ident
        old_deg = 0
        count = 0
        all_es = []
        while v.degree > old_deg:
            old_deg = v.degree
            es, seg, pie = self._find_edges(v.x, v.y, v.z, seg, pie, 
                                            v.edges+all_es)
            all_es.extend(es)
        v.add_edges(all_es, self)
        return v, seg, pie

    def vertexify(self, seg_arr=None, pie_arr=None):
        if seg_arr is None and pie_arr is None:
             seg_arr, pie_arr = self.array_rep()
        i = 0
        # allows for handling of newly created segments
        # on the fly
        while i < self.num_segs:
            segment = self[i]
            start = segment[0]
            if start.vertex is None:
                v, seg_arr, pie_arr = self._seed_edge_search(start, seg_arr, 
                                                             pie_arr)
                self._vertices[v.ident] = v
            end = segment[-1]
            if end.vertex is None:
                v, seg_arr, pie_arr = self._seed_edge_search(end, seg_arr, 
                                                             pie_arr)
                self._vertices[v.ident] = v
            i += 1

    def prune_vertices(self, minlen=10):
        for k in self.vertices.keys():
            v = self.vertices[k]
            for e in v.edges[:]:
                edge = self.segment_id(e[0])
                if edge.length() < minlen:
                    v.remove_edge(e)
                    edge[0].vertex = None
                    edge[-1].vertex = None
            if v.degree == 0:
                del self.vertices[k]

    def verify_vertices(self, img=None, imgpath=None, n=10, deg=0, window=20,
                        tail=10):
        if img is None and imgpath is None:
            raise IOError('one of img or imgpath is required')
        elif img is None:
            img = tiff.imread(imgpath)
        if deg > 0:
            vs = filter(lambda x: x.degree > deg, self._vertices.values())
        else:
            vs = self._vertices.values()
        vs_look = random.sample(vs, n)
        for v in vs_look:
            x, y, z = v.xyz
            img_min = img.min()
            img_max = img.max()
            im = img[max(z-window, 0):z+window, y-window:y+window,
                     x-window:x+window]
            if 0 not in im.shape:
                fig = plt.figure(figsize=(4,10))
                im_xyax = fig.add_subplot(3,1,1)
                im_h1ax = fig.add_subplot(3,1,2)
                im_h2ax = fig.add_subplot(3,1,3)
                im_xy = im.mean(axis=0)
                im_h1 = im.mean(axis=1)
                im_h2 = im.mean(axis=2)
                im_xyax.imshow(im_xy, vmin=img_min, vmax=img_max)
                im_h1ax.imshow(im_h1, vmin=img_min, vmax=img_max)
                im_h2ax.imshow(im_h2, vmin=img_min, vmax=img_max)
                print 'v',x, y, z, v.ident
                for e in v.edges:
                    s = self.segment_id(e[0])
                    p = s.piece_id(e[1])
                    xp, yp, zp = p.xyz
                    print 'e1',xp, yp, zp, e
                    xp, yp, zp = (xp - (x - window), yp - (y - window), 
                                  zp - (z - window))
                    print 'e2',xp, yp, zp
                    if p.ident == s[0].ident:
                        ps = s[1:1+tail]
                    elif p.ident == s[-1].ident:
                        ps = s[-tail-1:-1]
                    pxyzs = []
                    for piece in ps:
                        xps, yps, zps = piece.xyz
                        pxyz = (xps - (x - window), yps - (y - window), 
                                zps - (z - window))
                        pxyzs.append(pxyz)
                    pxyzs = np.array(pxyzs)
                    im_xyax.plot(pxyzs[:,0], pxyzs[:,1], 'bo')
                    im_h1ax.plot(pxyzs[:,0], pxyzs[:,2], 'bo')
                    im_h2ax.plot(pxyzs[:,1], pxyzs[:,2], 'bo')
                    im_xyax.plot(xp, yp, 'go')
                    im_h1ax.plot(xp, zp, 'go')
                    im_h2ax.plot(yp, zp, 'go')
                im_xyax.plot(window, window, 'ro')
                im_h1ax.plot(window, window, 'ro')
                im_h2ax.plot(window, window, 'ro')
                fig.suptitle('degree '+str(v.degree)+'; '+str(v.ident)+':'
                             +str(v.edges))
        plt.show()
        
    def check_graph_soundness(self):
        edge_errors = []
        seg_errors = []
        dup_errors = []
        for v in self.vertices.values():
            for e in v.edges:
                try:
                    p = self.segment_id(e[0]).piece_id(e[1])
                except Exception:
                    seg_errors.append(e)
                else:
                    if p.vertex != v.ident:
                        edge_errors.append((v, e, p.vertex, v.ident))
                        if e in self.vertices[p.vertex].edges:
                            dup_errors.append((e, self.vertices[p.vertex], v))
                
        return edge_errors, seg_errors, dup_errors
            
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
    
    def __init__(self, path=None, segments=None, mpp=None, 
                 cylinder=True, ident=None):
        """ must have one of segments or path or both """
        if mpp is None:
            self.mpp = np.ones((1, 3))
        else:
            self.mpp = np.array(mpp).reshape(1,3)
        self._vertices = {}
        if path is None and segments is not None:
            super(SWC, self).__init__(segments, ident)
        elif path is not None:
            self._define_by_path(path, cylinder, segments, ident)
        
    def _define_by_segments(self, segments):
        self.segs = segments
        self.num_segs = len(segments) 

    def _sort_into_segs(self, sdict, pid_dict, par_dict):
        self.segs = []
        for s in sdict.keys():
            pieces = sdict[s]
            ids = pid_dict[s]
            pars = par_dict[s]
            top_parent_set = pars.difference(ids)
            top_par = top_parent_set.pop()
            assert len(top_parent_set) == 0
            par = top_par
            ordered = []
            while pieces.get(par, False):
                entry = pieces[par]
                par = entry.ident
                ordered.append(entry)
            seg = Segment(self.mpp, pieces=np.array(ordered))
            self.segs.append(seg)
            self.num_segs += 1

    def _define_by_eswc_path(self, eswc_text, cylinder):
        seg_dict = {}
        pid_dict = {}
        par_dict = {}
        for entry in eswc_text:
            if entry[0] == '#':
                pass
            else:
                swcent = Piece.from_string(entry, cylinder)
                # print swcent.seg
                try:
                    piece_dict = seg_dict[swcent.seg]
                    id_set = pid_dict[swcent.seg]
                    p_set = par_dict[swcent.seg]
                    try: 
                        e = piece_dict[swcent.par]
                        print 'damn it'
                    except:
                        pass
                    piece_dict[swcent.par] = swcent
                    id_set.add(swcent.ident)
                    p_set.add(swcent.par)
                except KeyError:
                    seg_dict[swcent.seg] = {}
                    pid_dict[swcent.seg] = set()
                    par_dict[swcent.seg] = set()
                    seg_dict[swcent.seg][swcent.par] = swcent
                    pid_dict[swcent.seg].add(swcent.ident)
                    par_dict[swcent.seg].add(swcent.par)
        self._sort_into_segs(seg_dict, pid_dict, par_dict)
        
    def _define_by_swc_path(self, swc_text, cylinder):
        warnings.warn('got swc (not eswc) to represent: THE SEGMENT NUMBERS '
                      'WILL NOT BE THE SAME AS IN VAA3D')
        dict_all = {}
        entries = 0
        for entry in swc_text:
            if entry[0] == '#':
                pass
            else:
                entries += 1
                swcent = Piece.from_string(entry, cylinder)
                if swcent.par == -1:
                    s = Segment(self.mpp, pieces=np.array([swcent]))
                    self.segs.append(s)
                    self.num_segs += 1
                else:
                    try:
                        here = dict_all[swcent.par]
                        s = Segment(self.mpp, pieces=np.array([swcent]))
                        self.segs.append(s)
                        self.num_segs += 1
                        if here is not False:
                            here.par = -1
                            h = Segment(self.mpp, pieces=np.array([here]))
                            self.segs.append(h)
                            self.num_segs += 1
                            dict_all[swcent.par] = False
                            swcent.par = -1
                    except KeyError:
                        dict_all[swcent.par] = swcent
        for i, segment in enumerate(self.segs):
            segment[0].seg = segment.ident
            most_parentest = segment[0]
            self.segs[i] = _delve_to_children(dict_all, most_parentest, 
                                              segment)

    def _define_by_path(self, path, cylinder, segs, ident):
        if ident is None:
            self.ident = path
        if segs is None:
            self.segs = []
            self.num_segs = 0
        else:
            self.segs = segs
            self.num_segs = len(segs)
        with open(path, 'rb') as swc_text:
            if path.split('.')[-1] == 'eswc':
                self._define_by_eswc_path(swc_text, cylinder)
            else:
                self._define_by_swc_path(swc_text, cylinder)

    @classmethod
    def from_vida_mat(cls, path, mpp, buff=70):
        from scipy.io import loadmat
        vmat = loadmat(path)
        v = vmat['vectorizedStructure']
        strands = v['Strands'][0,0].T
        verts_xyz = v['Vertices'][0,0]['AllVerts'][0,0]
        verts_rad = v['Vertices'][0,0]['AllRadii'][0,0]
        pid = 0
        segs = []
        for s in strands:
            # matlab indexing begins at 1, python begins at zero
            inds = s['StartToEndIndices'][0][0] - 1 
            ps = []
            n = len(inds)
            for i, ind in enumerate(inds):
                pid += 1
                y, x, z = verts_xyz[ind] - 70
                if i == n - 1:
                    par = -1
                else:
                    par = pid + 1
                rad = verts_rad[ind]
                atts = (pid, 2, x, y, z, rad, par)
                ps.append(Piece(atts, True))
            segs.append(Segment(mpp, pieces=ps))
        return SWC(segments=segs, microns_perpixel=mpp)

    def __getitem__(self, key):
        return self.segs[key]

    def __len__(self):
        return self.num_segs

    def __iter__(self):
        return iter(self.segs)

    def filter(self, function):
        new_segs = filter(function, self)
        return SWC(segments=new_segs, microns_perpixel=self.mpp)

    def write_subset(self, fname, func, headers=True):
        new_swc = self.filter(func)
        new_swc.write(fname)
        
class Segment(object): 

    _seg_counter = 0
    
    def __init__(self, mpp, ident=None, pieces=None):
        self.mpp = mpp
        if pieces is None:
            self._pieces = np.array([])
            self._num_pieces = 0
        else:
            self._pieces = np.array(pieces)
            self._num_pieces = len(pieces)
            if self._pieces[0].seg == -2:
                # no identity from pieces
                if ident is None:
                    Segment._seg_counter += 1
                    self.ident = Segment._seg_counter
                else:
                    self.ident = ident
                for p in self._pieces:
                    p.seg = self.ident
            else:
                self.ident = self._pieces[0].seg

    def __repr__(self):
        # rep = '## segment '+str(self.ident)+' ##\n'
        rep = ''
        for piece in self._pieces:
            rep += str(piece)
        return rep
        
    def __getitem__(self, key):
        return self._pieces[key]

    def __iter__(self):
        return iter(self._pieces)
    
    def __len__(self):
        return self._num_pieces

    def piece_id(self, pid, index=False):
        for i, p in enumerate(self._pieces):
            if p.ident == pid:
                ret = p
                if index:
                    ret = (p, i)
                return ret
        raise Exception('the piece ID '+str(pid)+' is not in Segment '
                        +str(self.ident))

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

    @memoize
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

    def _norm_vecs(self, vecs, axis=1):
        return np.divide(vecs, vecs.sum(axis).reshape(vecs.shape[0], 1))

    @memoize
    def avg_radius_corrected(self, weighted=True):
        dxyzs = self._xyz_diffs() # give directionality vectors
        
        norm_dxyzs = self._norm_vecs(np.abs(dxyzs))
        units = np.ones(dxyzs.shape)
        renorm_dxyzs = self._norm_vecs(units - norm_dxyzs)

        renorm_dxyzs_anis = np.multiply(renorm_dxyzs, self.mpp)
        midrads = np.convolve(self.rads, np.ones((2)) / 2., mode='valid')
        midrads = midrads.reshape(midrads.shape[0], 1)
        rads = np.multiply(renorm_dxyzs_anis, midrads)
        rads = rads.sum(1)
        lxyzs = self._piece_lens(dxyzs)
        avgrad = np.average(rads, axis=0, weights=lxyzs)
        return avgrad

    @memoize
    def avg_radius(self, weighted=False):
        if weighted:
            dxyzs = self._xyz_diffs()
            lxyzs = self._piece_lens(dxyzs)
            avgrad = np.average(self.rads[1:], axis=0, weights=lxyzs)
        else:
            avgrad = np.mean(self.rads)
        return avgrad * self.mpp.mean()
            
    @memoize
    def avg_direction(self):
        """ length weighted average direction of segment """
        diff_xyzs = self._xyz_diffs()
        length_xyzs = self._piece_lens(diff_xyzs)
        return np.average(diff_xyzs, axis=0, weights=length_xyzs)

    @memoize
    def cumu_direction(self):
        diff_xyzs = self._xyz_diffs()
        length_xyzs = self._piece_lens(diff_xyzs)
        return

    @memoize
    def length(self):
        """ length of segment, calculated going point to point """
        diff_xyzs = self._xyz_diffs()
        return self._piece_lens(diff_xyzs).sum() * self.mpp

    @memoize
    def crow_length(self):
        """ length of segment as crow flies, from first to last piece """
        return np.sqrt(np.sum((np.array(self[0].xyz) 
                               - np.array(self[-1].xyz))**2)) * self.mpp
    
    @memoize
    def crow_direction(self):
        """ directionality of segment, taken from first to last piece only """
        dir_ = np.array(self[-1].xyz) - np.array(self[0].xyz)
        if dir_[-1] < 0:
            dir_ = dir_ * -1
        return dir_

    def split(self, p_id=None, p_num=None):
        if p_id is None and p_num is not None:
            i_of = p_num
        elif p_id is not None and p_num is None:
            i_of = self.piece_id(p_id, True)[1]
        else:
            raise Exception('split takes either piece_id or piece_num, not '
                            'both or neither')
        new_seg = Segment(self.mpp, pieces=self._pieces[i_of:])
        self._pieces = self._pieces[:i_of]
        self._num_pieces = len(self._pieces) 
        return new_seg
    
    def array_rep(self, seg_arr=None, pie_arr=None):
        if seg_arr is None and pie_arr is None:
            coords = self.xyzs
            dims = coords.max(axis=0)[::-1]
            seg_arr = np.zeros(dims, dtype=list)
            pie_arr = np.zeros(dims, dtype=list)
        for piece in self:
            xyz = piece.xyz
            r_zyx = np.around(xyz)[::-1]
            if seg_arr[r_zyx[0], r_zyx[1], r_zyx[2]] != 0:
                seg_arr[r_zyx[0], r_zyx[1], r_zyx[2]].append(self.ident)
                pie_arr[r_zyx[0], r_zyx[1], r_zyx[2]].append(piece.ident)
            else:
                seg_arr[r_zyx[0], r_zyx[1], r_zyx[2]] = [self.ident]
                pie_arr[r_zyx[0], r_zyx[1], r_zyx[2]] = [piece.ident]
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
        self.vertex = None
        if len(attributes) > 7:
            self.seg = int(attributes[7])
            self.layer = int(attributes[8])
            self.feature = int(attributes[9])
        else:
            self.seg = -2
            self.layer = -2
            self.feature = -2

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

    def __repr__(self):
        if self.feature > -1:
            seg = self.seg
            layer = self.layer
            feat = self.feature
            enhanced = ' {} {} {}'.format(seg, layer, feat)
        else:
            enhanced = ''
        rep = '{} {} {} {} {} {} {}{}\n'.format(self.ident, self.struct, self.x,
                                              self.y, self.z, self.rad, 
                                              self.par, enhanced)
        return rep

    @property
    def xyz(self):
        return (self.x, self.y, self.z) 

    @property
    def attributes(self):
        return self.ident, self.struct, self.x, self.y, self.z, self.rad, \
            self.par, self.seg


class Vertex(object):

    _vertex_counter = 0

    def __init__(self, x, y, z, r, edges):
        Vertex._vertex_counter += 1
        self.ident = Vertex._vertex_counter
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
        e1 = swc.segment_id(edges[0][0]).piece_id(edges[0][1])
        e1.vertex = self.ident
        v = cls(e1.x, e1.y, e1.z, e1.rad, edges[0])
        return v.add_edges(edges[1:], swc)

    def update_edge(self, new_edge, old_edge):
            i = self.edges.index(old_edge)
            self.edges[i] = new_edge

    def add_edge_bypiece(self, piece):
        piece.vertex = self.ident
        self.add_edge(piece.x, piece.y, piece.z, piece.rad, 
                      (piece.seg, piece.ident))

    def remove_edge(self, e):
        self.edges.remove(e)
        self.degree -= 1        

    def add_edge(self, x, y, z, r, e):
        self.x = (self.degree*self.x + x) / (self.degree + 1.)
        self.y = (self.degree*self.y + y) / (self.degree + 1.)
        self.z = (self.degree*self.z + z) / (self.degree + 1.)
        self.rad = max(self.rad, r)
        self.edges.append(e)
        self.degree = self.degree + 1

    def add_edges(self, es, swc):
        for e in es:
            if SEGS_ORDERED_AND_COMPLETE:
                p = swc[e[0] - 1].piece_id(e[1])
            else:
                p = swc.segment_id(e[0]).piece_id(e[1])
            if p.vertex is not None:
                print 'whoops'
            p.vertex = self.ident
            self.add_edge(p.x, p.y, p.z, p.rad, e)

    def __repr__(self):
        return ('<'+str(self.ident)+':'+str(self.xyz)+'|'
                +str(self.degree)+':'+str(self.edges)+'>')

    @property
    def xyz(self):
        return (self.x, self.y, self.z)
