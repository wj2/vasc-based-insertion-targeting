#!/usr/local/bin/python

import numpy as np
import tiff.tifffile as tiff
from swc import SWC
from os.path import splitext
from compare import compare_masked_to_source_imgs

def invert_mask(mask):
    return ~mask.astype(bool)

def crop_mask(mask, bound, im_shape):
    newmins = (np.around(bound['x'][0]), np.around(bound['y'][0]), 
               np.around(bound['z'][0]))
    print bound
    print newmins
    print im_shape
    print mask.shape
    cropped_mask = mask[-newmins[2]:-newmins[2]+im_shape[0], 
                        -newmins[1]:-newmins[1]+im_shape[1],
                        -newmins[0]:-newmins[0]+im_shape[2]]
    return cropped_mask

def find_bounds(vascswc=None, vascswcpath=None, cylinder=True):
    if vascswcpath is not None:
        vascswc = SWC(vascswcpath)
    i = 0
    for seg in vascswc:
        for piece in seg:
            if i == 0: 
                xmin = piece.get_dimmin('x')
                ymin = piece.get_dimmin('y')
                zmin = piece.get_dimmin('z')
                    
                xmax = piece.get_dimmax('x')
                ymax = piece.get_dimmax('y')
                zmax = piece.get_dimmax('z')
            
                margin = 0

                i += 1
            else: 
                xmin = min(xmin, piece.get_dimmin('x'))
                ymin = min(ymin, piece.get_dimmin('y'))
                zmin = min(zmin, piece.get_dimmin('z'))
                
                xmax = max(xmax, piece.get_dimmax('x'))
                ymax = max(ymax, piece.get_dimmax('y'))
                zmax = max(zmax, piece.get_dimmax('z'))

                margin = max(margin, piece.rad + 0.5)

    if cylinder:
        bounds = {'x':(xmin - margin, xmax + margin), 
                  'y':(ymin - margin, ymax + margin), 
                  'z':(zmin - margin, zmax + margin)}
    else: 
        bounds = {'x':(xmin, xmax), 'y':(ymin, ymax), 'z':(zmin, zmax)}
     
    return bounds

def resize_mask(swc=None, swcpath=None, cylinder=True, mask=None, 
                maskpath=None, imgpath=None, imshape=None):
    bounds = find_bounds(swc, swcpath, cylinder)
    if imgpath is None and imshape is None:
        raise IOError('need to be given either imgpath or imshape')
    elif imgpath is not None and imshape is None:
        imshape = tiff.imread(imgpath).shape
    if maskpath is None and mask is None:
        print 'cannot resize mask, no path given'
        print '---- orig img ----'
        x = -np.around(bounds['x'][0])
        print 'x : '+str(x)+' to '+str(x+imshape[2])
        y = -np.around(bounds['y'][0])
        print 'y : '+str(y)+' to '+str(y+imshape[1])
        z = -np.around(bounds['z'][0])
        print 'z : '+str(z)+' to '+str(z+imshape[0])
    else:
        if mask is None:
            mask = tiff.imread(maskpath)
        cropped_mask = crop_mask(mask, bounds, imshape)
        mask_name = None
        if mask is None:
            mask_name = splitext(maskpath)[0] + '-cropped.tif'
            tiff.imsave(mask_name, cropped_mask)
    return cropped_mask, mask_name

def mask_findings(imgpath, maskpath, swcpath, cylinder):
    
    bounds = find_bounds(swcpath, cylinder) 
    mask = tiff.imread(maskpath)
    inverted_mask = invert_mask(mask)
    img = tiff.imread(imgpath)
    cropvert_mask = crop_mask(inverted_mask, bounds, img.shape)

    masked_img = cropvert_mask * img

    compare_masked_to_source_imgs(masked_img, img)
    tiff.imsave(splitext(imgpath)[0] + '_masked.tif', masked_img)
    return
