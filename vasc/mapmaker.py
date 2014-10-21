
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
    y_off = np.ceil(probe.shape[1] / 2.)
    x_off = np.ceil(probe.shape[2] / 2.)
    y_b = y_off
    y_e = segid_map.shape[1] - y_off
    x_b = x_off
    x_e = segid_map.shape[2] - x_off
    maps = {}
    for c in expect_chars:
        maps[c] = np.empty((y_e-y_b, x_e-x_b))
    if probe.shape[0] < segid_map.shape[0]:
        z_lim = probe.shape[0] + 1
    else:
        z_lim = segid_map.shape[0] + 1
    for i, y in enumerate(xrange(y_b, y_e)):
        print 'row: '+str(i)
        for j, x in enumerate(xrange(x_b, x_e)):
            segid_col = segid_map[:z_lim, y-y_off:y+y_off, x-x_off:x+x_off]
            binary_col = binary_map[:z_lim, y-y_off:y+y_off, x-x_off:x+x_off]
            cols = {}
            cols['sid_map'] = segid_col
            cols['pid_map'] = segid_col # STAND IN, WE DO NOTHING WITH THIS NOW
            cols['binary_map'] = binary_col
            chars = characterize.characterize_insertion(cols, graph, 
                                                        percentile=90)
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

def make_maps(args):
    data_triplets = [(args.data[i], args.data[i+1], float(args.data[i+2])) 
                     for i in xrange(0, len(args.data), 3)]
    dims = [(args.probe_dims[i], args.probe_dims[i+1], args.probe_dims[i+2]) 
            for i in xrange(0, len(args.probe_dims), 3)]
    rots = number_to_angles(args.rotations)
    rots = [(r, 0, 0) for r in rots]
    # create probes for use later
    buffs = args.aoe_buffers
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
                for r in rots:
                    print 'doing: '+str(d)+':'+str(b)+':'+str(r)
                    probe = probes[d][b][r]
                    found_maps = all_valid_insertions(probe, graph, segid_map, 
                                                      binary_map)
                    maps[triplet][d][b][r] = found_maps

    return maps
