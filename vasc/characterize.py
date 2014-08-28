
import numpy as np

def characterize_insertion(cols, graph, radlarge=6, hthresh=.785, 
                           percentile=None):
    """ 
    find : 
        length intersected, volume intersected, large vessels hit, horizontal
        vessels hit, vertical vessels hit, length of vessels hit, large AND 
        vertical vessels, horizontal AND large vessels, distance from nearest
        large vessel
    and compose results into dictionary for return
    in cols: pid_map, sid_map
    """
    if percentile is not None:
        radlarge = np.percentile(map(lambda x: x.avg_radius(), graph), 
                                 percentile)
    chars = {}
    vessels_hit = set(cols['sid_map'].flatten()).difference(set([0]))
    pieces_hit = set(cols['pid_map'].flatten()).difference(set([0]))
    chars['num_vessels'] = len(vessels_hit)
    chars['vol_intersected'] = cols['binary_map'].sum() * graph.micsperpix
    
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

    print chars, radlarge

    return chars
