


def characterize_insertion(cols, graph, radlarge=6):
    """ 
    find : 
        length intersected, volume intersected, large vessels hit, horizontal
        vessels hit, vertical vessels hit, length of vessels hit, large AND 
        vertical vessels, horizontal AND large vessels, distance from nearest
        large vessel
    and compose results into dictionary for return
    in cols: pid_map, sid_map
    """
    chars = {}
    chars['num_vessels'] = len(set(cols['sid_map']).difference(set([0])))
    chars['vol_of_vessels'] = sum(cols['binary_map']) * graph.micsperpix

    return 
