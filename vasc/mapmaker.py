
import swc as swc
import tiff.tifffile as tiff
from probeplacer import create_probes
from util import *
import numpy as np
import cPickle as pickle
import characterize
import matplotlib.pyplot as plt

def all_valid_insertions(probe, graph, segid_map, binary_map, y_off, x_off,
                         radlarge):
    expect_chars = ['num_vessels', 'vol_intersected', 'large_vessels', 
                    'vert_vessels', 'horiz_vessels', 'vl_vessels', 
                    'vs_vessels', 'lenall_vessels']
    maps = {}
    y_b, y_e = (y_off, segid_map.shape[0] - y_off)
    x_b, x_e = (x_off, segid_map.shape[1] - x_off)
    py, py_off = probe.shape[0], int(np.floor(probe.shape[0] / 2.)) 
    px, px_off = probe.shape[1], int(np.floor(probe.shape[1] / 2.))
    for c in expect_chars:
        maps[c] = np.empty((1, y_e-y_b, x_e-x_b))
    # if probe.shape[0] < segid_map.shape[0]:
    #     z_lim = probe.shape[0] + 1
    # else:
    #     z_lim = segid_map.shape[0] + 1
    for i, y in enumerate(xrange(y_b, y_e)):
    # for i, y in enumerate(xrange(y_b, y_b + 20)):
        print 'row: '+str(i)
        for j, x in enumerate(xrange(x_b, x_e)):
        # for j, x in enumerate(xrange(x_b, x_b + 20)):
            segid_ls = segid_map[y-py_off:y-py_off+py, x-px_off:x-px_off+px]
            binary_l = binary_map[y-py_off:y-py_off+py, x-px_off:x-px_off+px]
            assert probe.shape == segid_ls.shape
            sid_map = np.concatenate(segid_ls[probe > 0])
            sid_map = np.unique(sid_map)
            binary_layer = binary_l * probe
            chars = characterize.do_stats(sid_map, binary_layer, graph, 
                                          radlarge=radlarge)
            for c in expect_chars:
                maps[c][0, i, j] = chars[c]
    return maps
            
def number_to_angles(n):
    angle = 180. / n
    angles = []
    curr = 0
    while curr < 180:
        angles.append(curr)
        curr += angle
    return angles

def flatten_uval_map(uvmap):
    newarr = np.empty((uvmap.shape[1], uvmap.shape[2]), dtype=object)
    for y in xrange(uvmap.shape[1]):
        for x in xrange(uvmap.shape[2]):
            vs = np.unique(uvmap[:, y, x])
            if vs[0] == 0:
                vs = vs[1:]
            newarr[y, x] = vs
    return newarr

def process_triplets(data):
    data_triplets = []
    for i in xrange(0, len(data), 5):
        trip_mpps = tuple((float(x) for x in data[i+2:i+5]))
        trip = (data[i], data[i+1], trip_mpps)
        data_triplets.append(trip)
    return data_triplets

def make_maps(data, probe_dims, rotations, aoe_buffers, perc=None):
    # data_triplets = [(data[i], data[i+1], data[i+2:i+5]) 
    #                  for i in xrange(0, len(data), 5)]
    data_triplets = process_triplets(data)
    print data_triplets
    dims = [(probe_dims[i], probe_dims[i+1], probe_dims[i+2]) 
            for i in xrange(0, len(probe_dims), 3)]
    rots = number_to_angles(rotations)
    rots = [(r, 0, 0) for r in rots]
    # create probes for use later
    buffs = aoe_buffers
    maps = {}
    for triplet in data_triplets:
        maps[triplet] = {}
        swc_path, segid_path, mpp = triplet
        probes, y_off, x_off = create_probes(dims, rots, buffs, mpp=mpp)
        graph = swc.SWC(path=swc_path, mpp=mpp)
        if perc is not None:
            radlarge = np.percentile(map(lambda x: x.avg_radius(), graph), 
                                     perc)
        else:
            radlarge = 6
        print radlarge
        # graph = graph.vertexify()
        # es = graph.check_graph_soundness()
        # assert len(es[0]) == 0 and len(es[1]) == 0 and len(es[2]) == 0
        segid_map = tiff.imread(segid_path)
        binary_map = make_binary_map(segid_map)
        for d in dims:
            maps[triplet][d] = {}
            for b in buffs:
                maps[triplet][d][b] = {}
                # flatten arrays here
                flat_binary = binary_map[:d[0]+b].sum(0)
                flat_segid = flatten_uval_map(segid_map[:d[0]+b])
                for r in rots:
                    print 'doing: '+str(d)+':'+str(b)+':'+str(r)
                    probe = probes[d][b][r][0, :, :]
                    found_maps = all_valid_insertions(probe, graph, flat_segid, 
                                                      flat_binary, y_off, 
                                                      x_off, radlarge=radlarge)
                    maps[triplet][d][b][r] = found_maps

    return maps

def combine_angles_helper(rots_dict, func):
    newdict = {}
    maxval = len(rots_dict.values()) - 1 
    for i, rot in enumerate(rots_dict.values()):
        for char in rot.keys():
            if i == 0:
                newdict[char] = rot[char]
            else:
                newdict[char] = np.vstack((newdict[char], rot[char]))
                if i == maxval:
                    newdict[char] = func(newdict[char], axis=0)
    return newdict

def combine_angles(maps, func):
    newdict = {}
    for trip in maps.keys():
        newdict[trip] = {}
        for dim in maps[trip].keys():
            newdict[trip][dim] = {}
            for buff in maps[trip][dim].keys():
                combined = combine_angles_helper(maps[trip][dim][buff], func)
                newdict[trip][dim][buff] = combined
    return newdict

def get_minmax(dimdict, buff=0):
    mmv = {}
    for char in dimdict[dimdict.keys()[0]][buff].keys():
        for i, dim in enumerate(dimdict.keys()):
            dmin = np.min(dimdict[dim][buff][char])
            dmax = np.max(dimdict[dim][buff][char])
            if i == 0:
                vmin = dmin
                vmax = dmax
            else:
                vmin = min(dmin, vmin)
                vmax = max(dmax, vmax)
        mmv[char] = (vmin, vmax)
    return mmv
            

def make_dimplots(mapdict, anglefunc=np.min, buff=0, name=''):
    mapdict = combine_angles(mapdict, anglefunc)
    for dim in mapdict.keys():
        plot_probesizes(mapdict[dim], buff)
        plt.suptitle(dim)
    plt.savefig(name+'map.pdf', bbox_inches='tight')
    plt.show()

def plot_probesizes(dimdict, buff=0):
    fig = plt.figure()
    rows = len(dimdict.keys()) + 1
    cols = len(dimdict[dimdict.keys()[0]][buff].keys())
    probesizefunc = lambda x: x[1]
    mmv = get_minmax(dimdict, buff)
    ddkeys = sorted(dimdict.keys(), key=probesizefunc)
    for i, d in enumerate(ddkeys):
        chardict = dimdict[d][buff]
        plot_probesizemap(d, chardict, fig, rows, cols, i*cols + 1, mmv)
    plot_probesizeplot(dimdict, probesizefunc, np.mean, fig=fig, rows=rows,
                       cols=cols, num=(i+1)*cols + 1)

def plot_probesizemap(psize, chardict, fig, rows, cols, num, mmv):
    for i, char in enumerate(chardict.keys()):
        vmin, vmax = mmv[char]
        cax = fig.add_subplot(rows, cols, num + i)
        im = cax.imshow(chardict[char], vmin=vmin, vmax=vmax)
        assert chardict[char].min() >= vmin
        assert chardict[char].max() <= vmax
        if num == 1:
            plt.colorbar(im)
            cax.set_title(char)
        if i == 0:
            cax.set_ylabel(psize)

def plot_probesizeplot(dimdict, sizefunc, mapfunc, buff=0, fig=None, rows=None, 
                       cols=None, num=None):
    if fig is None or rows is None or cols is None or num is None:
        fig = plt.figure()
        rows = 1
        cols = len(dimdict[dimdict.keys()[0]][buff].keys())
        num = 1
    for i, char in enumerate(dimdict[dimdict.keys()[0]][buff].keys()):
        plist = []
        mlist = []
        mlist_error = []
        for d in dimdict.keys():
            cmap = dimdict[d][buff][char]
            plist.append(sizefunc(d))
            mlist.append(mapfunc(cmap))
            mlist_error.append(np.std(cmap))
        cplot = fig.add_subplot(rows, cols, num+i)
        plist, mlist = np.array(plist), np.array(mlist)
        mlist_error = np.array(mlist_error)
        pargs = np.argsort(plist)
        cplot.plot(plist[pargs], mlist[pargs], '-o', markersize=2)
        cplot.errorbar(plist[pargs], mlist[pargs], yerr=mlist_error[pargs])
        if num == 1:
            cplot.set_title(char)
            
                         
