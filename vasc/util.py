
import numpy as np
import os 
import warnings
from tiff import tifffile as tiff
from scipy.ndimage import gaussian_filter

def decide_dim(d, d_p, constraint, side1, side2):
    if d - d_p/2. >= 0:
        d_b = d - d_p/2.
        pd_b = 0
    else:
        d_b = 0
        warnings.warn('probe truncated on '+side1+' side', RuntimeWarning)
        pd_b = - d-d_p/2.
    if d + d_p/2. <= constraint:
        d_e = d + d_p/2.
        pd_e = d_p
    else:
        d_e = constraint
        warnings.warn('probe truncated on '+side2+' side', RuntimeWarning)
        pd_e = - (d + d_p/2. - constraint)
    return pd_b, pd_e, d_b, d_e

def extract_column(x_b, px_b, x_e, px_e, y_b, py_b, y_e, py_e, z, z_end, probe,
                   **kwargs):
    cols = {}
    if z < 0:
        probe = probe[-z:z_end-z, py_b:py_e, px_b:px_e]        
        for key in kwargs.keys():
            col = kwargs[key][:z_end, y_b:y_e, x_b:x_e]
            cols[key] = col * probe
        
    else:
        probe = probe[:z_end, py_b:py_e, px_b:px_e]
        for key in kwargs.keys():
            col = kwargs[key][z:z_end, y_b:y_e, x_b:x_e]
            cols[key] = col * probe
    return cols, probe    

def create_cachename(args):
    combname = os.path.abspath(args.pre) + '___' + os.path.abspath(args.post)
    return combname.replace('/','l').replace('.','-')

def load_imgdir(path):
    files = os.listdir(path)
    files = filter(lambda x: os.path.splitext(x)[-1] == '.tif', files)
    for i,f in enumerate(sorted(files)):
        im = os.path.join(path, f)
        img = tiff.imread(im)
        if i == 0: 
            stack = np.empty((len(files), img.shape[0], img.shape[1]))
        stack[i] = img
    return img

def load_img(img_path, raw=False, saturate=None, gf=False, bits=8, 
             mask=False):
    if os.path.isdir(img_path):
        img = load_imgdir(img_path)
    elif os.path.isfile(img_path):
        img = tiff.imread(img_path)
    else:
        raise IOError('img path does not exist or something')
    if not raw:
        for layer in img:
            if saturate is not None:
                lmax = np.around(np.percentile(layer, 100 - saturate))
                layer[layer >= lmax] = lmax
            else:
                lmax = float(layer.max())
            lmin = float(layer.min())
            lmax = lmax - lmin
            layer = (layer - lmin) / lmax
            layer = layer * 2**bits          
        if gf:
            img = gaussian_filter(img, 2)
        img = np.around(img)
        img = img.astype('uint'+str(bits))
    return img

def load_mask(mask_path, swc, shape, binary=False):
    mask = load_img(mask_path, raw=True)
    print 'howdy',mask.shape
    if binary:
        mask[mask == mask.max()] = 1
    if shape != mask.shape:
        mask = subtractfound.resize_mask(swc=swc, mask=mask, imshape=shape)
    print 'heyho',mask.shape
    return mask


def collapse_stack(stack, collapse):
    if collapse > 1: 
        z_dim, y_dim, x_dim = stack.shape
        z_dim = z_dim / collapse
        new_container = np.zeros((z_dim, y_dim, x_dim))
        for i in xrange(z_dim):
            x = i * collapse
            new_container[i] = np.mean(stack[x:x+z_dim], axis=0)
    else:
        new_container = stack
    return new_container

def rotate_in_plane(xy1, xyc, ang, degrees=True):
    if degrees:
        ang = ang *  np.pi / 180
    xy1 = np.array(xy1)
    xyc = np.array(xyc)
    x, y = xy1 - xyc
    x2 = x * np.cos(ang) - y * np.sin(ang)
    y2 = x * np.sin(ang) + y * np.cos(ang)
    return x2, y2
