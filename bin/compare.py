#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
from scipy.misc import imrotate
from scipy import ndimage
from scipy.stats.stats import pearsonr

import tiff.tifffile as tiff

def make_green_alpha_scale_cm():
    greenalphadict = {'red' : [(0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)],
               'green' : [(0.0, 0.0, 0.0),
                          (1.0, 1.0, 1.0)],
               'blue' : [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)],
               'alpha' : [(0.0, 0.0, 0.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 1.0, 1.0)]}
    greenalphascale = color.LinearSegmentedColormap('greenalphascale', 
                                                  greenalphadict)
    return greenalphascale

def _unit_vector(a):
    return a / np.linalg.norm(a)

def angle_between(a, b, degrees=False, direction=False):
    a = _unit_vector(a)
    b = _unit_vector(b)
    angle = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if direction:
        dir_ = a[0]*b[1] - a[1]*b[0]
        if dir_ > 0: dir_ = 1
        else: dir_ = -1
        angle = dir_ * angle
    if degrees:
        angle = angle * 180 / np.pi
    return angle

def clean_rotate(line, ang, interp='nearest'):
    rotated = imrotate(line, ang, interp=interp)
    # get rid of artifacts
    rotated[rotated < rotated.max()] = 0
    rotated[rotated == rotated.max()] = 1
    return rotated

def make_line(l, angle=None):
    line = np.zeros((l, l))
    line[l/2, :] = 1
    if angle != None:
        line = clean_rotate(line, angle)
    return line

def piece_profile(v, crds, rad, add=10, unit=None):
    if unit == None:
        unit = np.array([1,0,0])
    o = np.array([-v[1], v[0], 0])
    ang = angle_between(o, unit, True, True)
    len_ = np.around((rad * 2.) + add)
    x_b = np.around(crds[0] - (len_ / 2))
    x_e = crds[0] + (len_ / 2)
    y_b = crds[1] - (len_ / 2)
    y_e = crds[1] + (len_ / 2)
    if not (x_b < 0 or x_e >= img.shape[2] or 
            y_b < 0 or y_e >= img.shape[1]):
        section = img[crds[2], y_b:y_e, x_b:x_e]
        
        prof, section_r = line_profile_radius(section, ang, len_)
        prof_rad = profile_rad(prof)
    else:
        raise IndexError('section is out of the image')
    return prof_rad, prof, section_r
        

def filled_radius_line_profile(swc, img):
    img = tiff.imread(img)
    calc_rads = []
    fill_rads = []
    skipped = 0
    all_ = 0
    plotted = 0
    seen = 0
    for segment in swc:
        for i, piece in enumerate(segment[1:-1]):
            all_ += 1
            meanpiece_v = np.array(segment[i+2].xyz) - np.array(segment[i].xyz)
            try:
                prof_rad, prof, section_r = piece_profile(meanpiece_v, 
                                                          piece.xyz, piece.rad)
            except IndexError:
                skipped += 1
            else:
                if (np.random.uniform() < .00000000005 
                    and piece.rad - prof_rad > 6 
                    and seen < 10
                    and False):
                    seen += 1
                    pltitle = (str(segment.id)+':'+str(i+1)+':'+str(piece.id)
                               +' | image rotated '+str(seen))
                    plot_line_prof(section_r, make_line(section_r.shape[1]),
                                   prof, pltitle, prof_rad, piece.rad)

                plotted += 1 
                fill_rads.append(piece.rad)
                calc_rads.append(prof_rad)
    
    rad_corr = np.corrcoef(fill_rads, calc_rads)[0, 1]
    # print 'numpy correlation = ' + str(rad_corr)
    pcorr = pearsonr(fill_rads, calc_rads)
    print 'pearson r = '+str(pcorr[0])
    # print 'p-val     = '+str(pcorr[1])
    # print str(skipped) + ' skipped'
    # print str(plotted) + ' plotted'
    # print str(all_ - plotted - skipped) + ' excluded'
    print '-------- line profiles --------'
    print 'range : '+str(min(calc_rads))+','+str(max(calc_rads))
    print 'median: '+str(np.median(calc_rads))
    print 'mean  : '+str(np.mean(calc_rads))
    print 'std   : '+str(np.std(calc_rads))
    print '-------- vaa3d radius fill --------'
    print 'range : '+str(min(fill_rads))+','+str(max(fill_rads))
    print 'median: '+str(np.median(fill_rads))
    print 'mean  : '+str(np.mean(fill_rads))
    print 'std   : '+str(np.std(fill_rads))
    plot_radii(fill_rads, calc_rads)

def profile_rad(p, n=3, avg=True):
    p = np.array(p)
    dp = np.diff(p)
    if avg:
        window = np.ones((n,)) / n
        dpcm = np.convolve(dp, window, mode='valid')
        dpa_max = dpcm[:2*dpcm.shape[0]/3].argmax()
        dpa_min = dpcm[dpa_max:].argmin() + dpa_max
        rad = (dpa_min - dpa_max + 1) / 2.
    else:
        dpa_max = dp[:2*dp.shape[0]/3].argmax()
        dpa_min = dp[dpa_max:].argmin() + dpa_max
        rad = (dpa_min - dpa_max + 1) / 2.
    return rad

def line_profile_radius(img, ang, l):
    rot_img = ndimage.rotate(img, -ang, mode='nearest', reshape=False)
    l = rot_img.shape[0]
    return rot_img[l/2, :].astype(np.int16), rot_img

def line_profile_radius_profrotate(img, ang, l):
    line = make_line(l, ang)
    prof_2d = img * line
    return prof_2d.T[prof_2d.T.nonzero()].astype(np.int16), img

def plot_line_prof(sect, overlay, profile, title=None, prof_rad=None,
                   piece_rad=None, ang=None):
    fig = plt.figure()
    if title != None:
        fig.suptitle(title)
    im_plot = fig.add_subplot(2, 1, 1)
    if ang != None:
        im_plot.set_title('rotated '+str(ang)+' degrees')
    im_plot.imshow(sect, interpolation='none')
    gcm = make_green_alpha_scale_cm()
    im_plot.imshow(overlay, gcm, alpha=.50, interpolation='none')
    prof_plot = fig.add_subplot(2, 1, 2)
    prof_title = ''
    if prof_rad != None:
        prof_title += 'profile-judged radius: '+str(prof_rad)
    if prof_rad != None and piece_rad != None:
        prof_title += ' | '
    if piece_rad != None:
        prof_title += 'vaa3d-judged radius: '+str(piece_rad)
    prof_plot.plot(profile)
    prof_plot.set_title(prof_title)
    plt.show()

def plot_radii(fill_rads, calc_rads):
    rfig = plt.figure()
    splot = rfig.add_subplot(111)
    splot.set_xlabel('vaa3d radius fill rads')
    splot.set_ylabel('line profile calculated rads')

    splot.plot(fill_rads, calc_rads, 'bo', markersize=2)
    rad_max = max(fill_rads + calc_rads)
    splot.axis([0, rad_max, 0, rad_max])

    splot.plot([0, rad_max], [0, rad_max], 'r--', label='y=x')
    coefs = np.polyfit(fill_rads, calc_rads, 1)
    fit_y = np.polyval(coefs, fill_rads)
    splot.plot(fill_rads, fit_y, 'b--', label='line of best fit')
    # hand, label = splot.get_legend_handles_labels()
    splot.legend(loc=2)

    plt.show()
    return
    

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
