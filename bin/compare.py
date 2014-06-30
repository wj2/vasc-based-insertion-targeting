#!/usr/local/bin/python

import numpy as np
import tiff.tifffile as tiff

def get_total_lum(img, threshold=0):
    return img[np.where(img >= threshold)].sum()

def compare_masked_to_source_imgs(masked, source):
    mask_mean = masked.mean()
    mask_median = np.median(masked)
    mask_std = masked.std()
    mask_thresh = mask_mean + mask_std
    mask_total = get_total_lum(masked, threshold=mask_thresh)

    sour_mean = source.mean()
    sour_median = np.median(source)
    sour_std = source.std()
    sour_thresh = sour_mean + sour_std
    sour_total = get_total_lum(source, threshold=mask_thresh)
    mask_sour_thresh = get_total_lum(masked, threshold=sour_thresh)

    lum_absorbed = sour_total - mask_total
    absorbed_bysourcetotal = lum_absorbed / float(sour_total)

    maskbysource = mask_sour_thresh / float(sour_total)
    table = [['', 'masked', 'source'],
             ['mean', mask_mean, sour_mean],
             ['median', mask_median, sour_median],
             ['std', mask_std, sour_std],
             ['total > thresh', mask_total, sour_total],
             ['masked / source', '', maskbysource],
             ['1 - masked / source', '', 1 - maskbysource],
             ['absorbed', '', lum_absorbed],
             ['absorbed / source_total', '', absorbed_bysourcetotal]]

    for row in table:
        print("{:>20} {:>20} {:>20}".format(*row))
    return

def compare_masked_to_source(masked, source):
    masked = tiff.imread(masked)
    source = tiff.imread(source)
    compare_masked_to_source_imgs(masked, source)
