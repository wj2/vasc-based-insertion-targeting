#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
from SWC import Piece

def read_and_pull(path, cylinder):
    entries = {}
    radii = np.array([])
    with open(path, 'rb') as swc:
        for entry in swc:
            entry = entry.strip()
            if entry[0] == '#':
                pass
            else:
                ent = Piece.from_string(entry, cylinder)
                entries[ent.ident] = ent
                radii = np.append(radii, ent.rad)
    
    return entries, radii

def radius_stats_and_plot(radii):
    radius = {}
    radius['mean'] = np.mean(radii)
    radius['median'] = np.median(radii)
    radius['std'] = np.std(radii)
    plt.hist(radii, bins=100)
    plt.show()

    return radius

def segment_vectors(swc_entries, swc_size): 
    for x in 
    
