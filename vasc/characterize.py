
import numpy as np

def do_stats(vessels_hit, binlayer, graph, radlarge=6, hthresh=.785):
    chars = {}
    chars['num_vessels'] = len(vessels_hit)
    chars['vol_intersected'] = binlayer.sum() * np.product(graph.mpp)
    
    segs = map(lambda x: graph.segment_id(x), vessels_hit)
    chars['large_vessels'] = len(filter(lambda x: x.avg_radius() >= radlarge, 
                                        segs))
    chars['vert_vessels'] = len(filter(lambda x: x.downwardness() < hthresh, 
                                       segs))
    chars['horiz_vessels'] = len(segs) - chars['vert_vessels']
    chars['vl_vessels'] = len(filter(lambda x: x.downwardness() < hthresh and
                                     x.avg_radius() >= radlarge, segs))
    chars['vs_vessels'] = chars['large_vessels'] - chars['vl_vessels']
    chars['lenall_vessels'] = reduce(lambda y, x: x.length() + y, segs, 0)
    # chars['lenhit_vessels'] = 
    return chars

def characterize_insertion(cols, graph, radlarge=6, hthresh=.785):
    """ 
    find : 
        length intersected, volume intersected, large vessels hit, horizontal
        vessels hit, vertical vessels hit, length of vessels hit, large AND 
        vertical vessels, horizontal AND large vessels, distance from nearest
        large vessel
    and compose results into dictionary for return
    in cols: pid_map, sid_map
    """
    # vessels_hit = set(cols['sid_map'].flatten()).difference(set([0]))
    vessels_hit = np.unique(cols['sid_map'])
    if vessels_hit[0] == 0:
        vessels_hit = vessels_hit[1:]
    binlayer = cols['binary_map'].sum(0)
    # pieces_hit = set(cols['pid_map'].flatten()).difference(set([0]))
    return do_stats(vessels_hit, binlayer, graph, radlarge, hthresh)
