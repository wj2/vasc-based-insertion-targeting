#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imrotate

import tiff.tifffile as tiff

def _unit_vector(a):
    return a / np.linalg.norm(a)

def angle_between(a, b, degrees=False):
    a = _unit_vector(a)
    b = _unit_vector(b)
    angle = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if degrees:
        angle = angle * 180 / np.pi
    return angle

def clean_rotate(line, ang, interp='nearest'):
    rotated = imrotate(line, ang, interp=interp)
    # get rid of artifacts
    rotated[rotated < rotated.max()] = 0
    rotated[rotated == rotated.max()] = 1
    return rotated

def make_line(l, angle):
    line = np.zeros((l, l))
    line[l/2, :] = 1
    return clean_rotate(line, angle)

def filled_radius_line_profile(swc, img):
    img = tiff.imread(img)
    unit_v = np.array([1,0,0])
    for segment in swc:
        diffs = segment._xyz_diffs()
        for i, piece in enumerate(segment[1:-1]):
            meanpiece_v = diffs[i:i+2].mean(axis=0)
            orthog_v = np.array([-meanpiece_v[1],
                                 meanpiece_v[0],
                                 0])
            print meanpiece_v, orthog_v
            # create line profile
            profile_dang = -angle_between(orthog_v, unit_v, True)
            profile_len = np.around((piece.rad * 2.) + 10)
            profile = make_line(profile_len, profile_dang)
            
            # multiply at correct location
            crds = piece.xyz
            x_b = crds[0] - (profile_len / 2)
            x_e = crds[0] + (profile_len / 2)
            y_b = crds[1] - (profile_len / 2)
            y_e = crds[1] + (profile_len / 2)
            print x_b, x_e
            print y_b, y_e
            if not (x_b < 0 or x_e >= img.shape[2] or 
                    y_b < 0 or y_e >= img.shape[1]):
                section = img[crds[2], y_b:y_e, x_b:x_e]
                print crds
                print piece.rad
                print profile_dang
                print section.shape, profile.shape
                prof = section * profile
                prof = prof.T[prof.T.nonzero()]
                plt.plot(prof)
            else:
                print 'skipped'


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
