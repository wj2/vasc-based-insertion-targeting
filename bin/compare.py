#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.gridspec as gs
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

def piece_profile(v, img, crds, rad, add=10, unit=None):
    if unit == None:
        unit = np.array([1,0,0])
    o = np.array([-v[1], v[0], 0])
    ang = angle_between(o, unit, True, True)
    len_ = np.around((rad * 2.) + add)
    z = np.around(crds[2])
    x_b = crds[0] - (len_ / 2)
    x_e = crds[0] + (len_ / 2)
    x = crds[0] - x_b
    y_b = crds[1] - (len_ / 2)
    y_e = crds[1] + (len_ / 2)
    y = crds[1] - y_b
    if not (x_b < 0 or x_e >= img.shape[2] or 
            y_b < 0 or y_e >= img.shape[1]):
        section = img[z, y_b:y_e, x_b:x_e]
        
        prof, section_r = line_profile_radius(section, ang, len_)
        prof_rad, l, r = profile_rad(prof)
    else:
        raise IndexError('section is out of the image')
    return prof_rad, (l, r), (x, y), prof, section_r
        

def filled_radius_line_profile(swc, img, random_display=False, dfp_thresh=6, 
                               rshow=5, main_display=True, lessthan=True,
                               absolute=True):
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
                prof_rad, lr, xy, prof, section_r = piece_profile(meanpiece_v, 
                                                                  img, 
                                                                  piece.xyz,
                                                                  piece.rad)
            except IndexError:
                skipped += 1
            else:
                diff = piece.rad - prof_rad
                if absolute: diff = np.abs(diff)
                if (random_display and np.random.uniform() < .05 
                    and ((lessthan and diff < dfp_thresh) or 
                         (not lessthan and diff > dfp_thresh))
                    and seen < rshow):
                    seen += 1
                    pltitle = (str(segment.ident)+':'+str(i+1)+':'
                               +str(piece.ident)+' | Figure '+str(seen))
                    plot_line_prof(section_r, make_line(section_r.shape[1]),
                                   prof, lr, xy, pltitle, prof_rad, piece.rad)

                plotted += 1 
                fill_rads.append(piece.rad)
                calc_rads.append(prof_rad)
    
    if main_display:
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
        dpa_min = dpa_min + 1
        dpa_max = dpa_max + 1
    else:
        dpa_max = dp[:2*dp.shape[0]/3].argmax()
        dpa_min = dp[dpa_max:].argmin() + dpa_max
        rad = (dpa_min - dpa_max + 1) / 2.
    return rad, dpa_max, dpa_min

def line_profile_radius(img, ang, l):
    rot_img = ndimage.rotate(img, -ang, mode='nearest', reshape=False)
    l = rot_img.shape[0]
    return rot_img[l/2, :].astype(np.int16), rot_img

def line_profile_radius_profrotate(img, ang, l):
    line = make_line(l, ang)
    prof_2d = img * line
    return prof_2d.T[prof_2d.T.nonzero()].astype(np.int16), img

def plot_line_prof(sect, overlay, profile, plr=None, xy=None, title=None, 
                   prof_rad=None, piece_rad=None, ang=None):
    fig = plt.figure(figsize=(6,8))
    spec = gs.GridSpec(4, 3)
    im_plot = plt.subplot(spec[:-1,:])
    prof_plot = plt.subplot(spec[-1,:], sharex=im_plot)
    if title != None:
        fig.suptitle(title)
    if ang != None:
        im_plot.set_title('rotated '+str(ang)+' degrees')
    im_plot.imshow(sect, interpolation='none')
    gcm = make_green_alpha_scale_cm()
    im_plot.imshow(overlay, gcm, alpha=.50, interpolation='none')
    prof_title = ''
    if prof_rad != None:
        prof_title += 'profile-judged radius: '+str(prof_rad)
    if prof_rad != None and piece_rad != None:
        prof_title += ' | '
    if piece_rad != None and xy != None:
        prof_title += 'vaa3d-judged radius: '+str(piece_rad)
        xy = (xy[0] - .5, xy[1] - .5)
        imcirc = plt.Circle(xy, piece_rad, color='g', fill=False,
                            label='one')
        im_plot.add_artist(imcirc)
    if plr != None:
        imrect = plt.Rectangle((plr[0], (sect.shape[0]/2) - .5), 
                               plr[1] - plr[0], 1, fill=False, color='g')
        im_plot.add_artist(imrect)
        ps = profile[plr[0]:plr[1]+1]
        prrect = plt.Rectangle((plr[0], min(ps)), plr[1] - plr[0], 
                               max(ps) - min(ps), fill=True, color='g', 
                               alpha=.75)
        prof_plot.add_artist(prrect)
    if plr != None and piece_rad != None:
        imcirc_l = plt.Line2D([0], [0], color='white', marker='o',
                              markeredgecolor='g', markerfacecolor='white', 
                              markersize=10)
        im_plot.legend((imcirc_l, imrect), ('vaa3d','line profile'),
                       numpoints=1)
        
    prof_plot.plot(profile)
    im_plot.set_title(prof_title)
    im_plot.axis([0, sect.shape[1]-1, 0, sect.shape[0]-1])
    plt.setp(im_plot.get_xticklabels(), visible=False)
    plt.setp(im_plot.get_yticklabels(), visible=False)
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
