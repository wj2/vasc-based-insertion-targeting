
import swc as swc
import tiff.tifffile as tiff
from probeplacer import create_probes
from util import *
import numpy as np
import cPickle as pickle
import characterize

def all_valid_insertions(probe, graph, segid_map, binary_map):
    expect_chars = ['num_vessels', 'vol_intersected', 'large_vessels', 
                    'vert_vessels', 'horiz_vessels', 'vl_vessels', 
                    'vs_vessels', 'lenall_vessels']
    # FACTOR OUT boundary calculation, take most stringent boundaries at 
    # beginning to ensure all maps have same dims
    y_off = int(np.ceil(probe.shape[0] / 2.))
    x_off = int(np.ceil(probe.shape[1] / 2.))
    y_b = y_off
    y_e = segid_map.shape[0] - y_off
    x_b = x_off
    x_e = segid_map.shape[1] - x_off
    print 'y',y_b, y_e
    print 'x',x_b, x_e
    maps = {}
    for c in expect_chars:
        maps[c] = np.empty((y_e-y_b, x_e-x_b))
    # if probe.shape[0] < segid_map.shape[0]:
    #     z_lim = probe.shape[0] + 1
    # else:
    #     z_lim = segid_map.shape[0] + 1
    for i, y in enumerate(xrange(y_b, y_e)):
    # for i, y in enumerate(xrange(y_b, y_b + 20)):
        print 'row: '+str(i)
        for j, x in enumerate(xrange(x_b, x_e)):
        # for j, x in enumerate(xrange(x_b, x_b + 20)):
            segid_ls = segid_map[y-y_off:y+y_off, x-x_off:x+x_off]
            binary_l = binary_map[y-y_off:y+y_off, x-x_off:x+x_off]

            assert probe.shape == segid_ls.shape
            sid_map = np.concatenate(segid_ls[probe > 0])
            sid_map = np.unique(sid_map)
            binary_layer = binary_l * probe
            chars = characterize.do_stats(sid_map, binary_layer, graph)
            for c in expect_chars:
                maps[c][i, j] = chars[c]
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

def make_maps(data, probe_dims, rotations, aoe_buffers):
    data_triplets = [(data[i], data[i+1], float(data[i+2])) 
                     for i in xrange(0, len(data), 3)]
    dims = [(probe_dims[i], probe_dims[i+1], probe_dims[i+2]) 
            for i in xrange(0, len(probe_dims), 3)]
    rots = number_to_angles(rotations)
    rots = [(r, 0, 0) for r in rots]
    # create probes for use later
    buffs = aoe_buffers
    probes = create_probes(dims, rots, buffs)
    maps = {}
    for triplet in data_triplets:
        maps[triplet] = {}
        swc_path, segid_path, mpp = triplet
        graph = swc.SWC(path=swc_path, microns_perpixel=mpp)
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
                                                      flat_binary)
                    maps[triplet][d][b][r] = found_maps

    return maps

def combine_angles(maps, func):
    for trip in maps.keys():
        for dim in trip.keys():
            for b in dim.keys():
                for i, rot in b.keys():
                    stack = np.dstack(b.values())
    
