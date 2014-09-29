
from subprocess import check_call
import argparse
import tempfile as tf
import tiff.tifffile as tiff
import time
import os
from config import VAA3D, VAA3D_DIR

"""
Tracing procedure

gaussian filter
(contrast enhancement?)
snake trace
radius fill
swc2mask
distance transform
snake trace
radius fill w/ reference to original

ouput: swc, swc2mask-segid

"""

def snake_trace(img_path):
    cmd = VAA3D 
    x_flag = '-x ' + os.path.join(VAA3D_DIR, 'bin/plugins/neuron_tracing/'
                                  'snake_tracing/libsnaketracing.so')
    f_flag = '-f snake_trace'
    i_flag = img_path
    img_name, img_ext = os.path.splitext(img_path)
    swc_out = img_name + '-traced.swc'
    o_flag = '-o ' + swc_out
    retcode = check_call([cmd, x_flag, f_flag, i_flag, o_flag])
    return swc_out

def radius_fill(swc_path, reference_img_path):
    cmd = VAA3D
    x_flag = '-x neuron_radius'
    f_flag = '-f neuron_radius'
    i_flag = '-i ' + reference_img_path + ' ' + swc_path
    swc_out = swc_path + '.out.swc'
    retcode = check_call([cmd, x_flag, f_flag, i_flag])
    return swc_out

def swc_to_mask(swc_path, segid=False):
    cmd = VAA3D
    i_flag = '-i ' + swc_path
    swc_name, swc_ext = os.path.splitext(swc_path)
    if segid:
        x_flag = '-x ' + os.path.join(VAA3D_DIR, 'bin/plugins/neuron_utilities'
                                      '/swc_to_maskimage_BRL/libswc2maskBRL.so')
        f_flag = '-f swc2maskBRL'
        tif_out = swc_name + '-segidmask.tif'
        o_flag = '-o ' + tif_out
    else:
        x_flag = '-x ' + os.path.join(VAA3D_DIR, 'bin/plugins/neuron_utilities'
                                      'swc_to_maskimage_cylinder_unit/'
                                      'libswc2mask.so')
        f_flag = '-f swc2mask'
        tif_out = swc_name + '-mask.tif'
        o_flag = '-o ' + tif_out
    retcode = check_call([cmd, x_flag, f_flag, i_flag, o_flag])
    # ADD IN CROPPING OF MASK
    return tif_out

def distance_transform(img_path):
    cmd = VAA3D
    x_flag = '-x gsdt'
    f_flag = '-f gsdt'
    i_flag = '-i ' + img_path
    p_flag = '-p .5 1 0 1'
    img_name, img_ext = os.path.splitext(img_path)
    tif_out = img_name + '-distancetransformed' + img_ext
    o_flag = '-o ' + tif_out
    retcode = check_call([cmd, x_flag, f_flag, i_flag, o_flag, p_flag])
    return tif_out

def gaussian_filter(img_path):
    cmd = VAA3D
    x_flag = '-x gaussian'
    f_flag = '-f gf'
    i_flag = '-i ' + img_path
    img_name, img_ext = os.path.splitext(img_path)
    tif_out = img_name + '-gf55512' + img_ext
    o_flag = '-o ' + tif_out
    p_flag = '-p 5 5 5 1 2'
    retcode = check_call([cmd, x_flag, f_flag, i_flag, o_flag, p_flag]) 
    return tif_out

# input is already gf'd and (maybe) contrast enhanced
def trace_vasc_vaa3d(img, mpp, gf=True):
    base_name = str(time.time())
    tmpdir = tf.gettempdir()
    orig_dir = os.getcwd()
    os.chdir(tmpdir)
    img_base_name = base_name+'_base.tif'
    tiff.imsave(img_base_name, img)
    if not gf:
        img_base_name = gaussian_filter(img_base_name)
    first_swc = snake_trace(img_base_name)
    first_swc = radius_fill(first_swc, img_base_name)
    first_swc_mask = swc_to_mask(first_swc)
    distance_mask = distance_transform(first_swc_mask)
    second_swc = snake_trace(distance_mask)
    second_swc = radius_fill(second_swc, img_base_name)
    second_swc_mask_segid = swc_to_mask(second_swc, segid=True)
    
    finished_swc = swc.SWC(path=second_swc, microns_perpixel=mpp)
    
    return finished_swc, tiff.imread(second_swc_mask_segid)
    
    

