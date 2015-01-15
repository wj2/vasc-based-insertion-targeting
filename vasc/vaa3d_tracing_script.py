
from subprocess import check_call
import argparse
import tempfile as tf
import tiff.tifffile as tiff
import time
import os
from config import VAA3D, VAA3D_DIR
import subtractfound
import swc
import cPickle as p
VAA3D = 'vaa3d'
VAA3D_DIR = '/local1/vaa3d/v3d_external/'
# VAA3D_DIR = '/Users/wjj/Applications/Vaa3d-mkspec/v3d_external/'
FIJI = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'

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

SNAKE_TRACE = 'bin/plugins/neuron_tracing/snake_tracing/libsnake_tracing.so'
# SNAKE_TRACE = 'bin/plugins/neuron_tracing/snake_tracing/libsnake_tracing_debug.dylib'
def snake_trace(img_path):
    cmd = [VAA3D]
    x_flag = ['-x', os.path.join(VAA3D_DIR, SNAKE_TRACE)]
    f_flag = ['-f', 'snake_trace']
    i_flag = ['-i', img_path]
    # img_name, img_ext = os.path.splitext(img_path)
    # swc_out = img_name + '-traced.swc'
    swc_out = img_path + '_snake.swc'
    o_flag = ['-o', swc_out]
    retcode = check_call(cmd+x_flag+f_flag+i_flag+o_flag)
    return swc_out

def radius_fill(swc_path, reference_img_path):
    cmd = [VAA3D]
    x_flag = ['-x', 'neuron_radius']
    f_flag = ['-f', 'neuron_radius']
    i_flag = ['-i', reference_img_path, swc_path]
    swc_out = swc_path + '.out.swc'
    retcode = check_call(cmd+x_flag+f_flag+i_flag)
    return swc_out

SWC2MASK_BRL = ('bin/plugins/neuron_utilities/swc_to_maskimage_BRL'
                '/libswc2maskBRL.so')
# SWC2MASK_BRL = ('bin/plugins/neuron_utilities/swc_to_maskimage_BRL/'
#                 'libswc2maskBRL_debug.dylib')
SWC2MASK = ('bin/plugins/neuron_utilities/swc_to_maskimage_cylinder_unit/'
            'libswc2mask.so')
# SWC2MASK = ('bin/plugins/neuron_utilities/swc_to_maskimage_cylinder_unit/'
#             'libswc2mask_debug.dylib')
def swc_to_mask(swc_path, shape, segid=False):
    cmd = [VAA3D]
    if segid:
        eswc_path = swc_to_eswc(swc_path)
        eswc_name, eswc_ext = os.path.splitext(eswc_path)
        i_flag = ['-i', eswc_path]
        x_flag = ['-x', os.path.join(VAA3D_DIR, SWC2MASK_BRL)]
        f_flag = ['-f','swc2maskBRL']
        v3d_out = eswc_name + '-segidmask.v3draw'
        o_flag = ['-o', v3d_out]
        flip = False
    else:
        swc_name, swc_ext = os.path.splitext(swc_path)
        i_flag = ['-i', swc_path]
        x_flag = ['-x', os.path.join(VAA3D_DIR, SWC2MASK)]
        f_flag = ['-f', 'swc2mask']
        tif_out = swc_name + '-mask.tif'
        o_flag = ['-o', tif_out]
        flip = True
    retcode = check_call(cmd+x_flag+f_flag+i_flag+o_flag)
    if segid:
        tif_out = v3draw_to_tif16(v3d_out)
    _, tif_out = subtractfound.resize_mask(swcpath=swc_path, maskpath=tif_out, 
                                           imshape=shape, flip=flip)
    return tif_out

prefix = os.path.dirname(os.path.realpath(__file__))
V3DRAW_CONV = prefix + '/brl_v3draw_convert.js'
def v3draw_to_tif16(imgpath):
    fname, ext = os.path.splitext(imgpath)
    outpath = fname + '.tif'
    cmd = FIJI+' --headless -macro '+V3DRAW_CONV+' '+imgpath+':'+outpath
    retcode = check_call(cmd.split(' '))
    return outpath

SWC2ESWC = ('bin/plugins/neuron_utilities/Enhanced_SWC_Format_Converter/'
            'libeswc_converter.so')
# SWC2ESWC = ('bin/plugins/neuron_utilities/Enhanced_SWC_Format_Converter/'
#             'libeswc_converter_debug.dylib')
def swc_to_eswc(swc_path):
    cmd = [VAA3D]
    x_flag = ['-x', os.path.join(VAA3D_DIR, SWC2ESWC)]
    f_flag = ['-f', 'swc_to_eswc']
    i_flag = ['-i', swc_path]
    swc_name, swc_ext = os.path.splitext(swc_path)
    eswc_path = swc_name + '.eswc'
    o_flag = ['-o', eswc_path]
    retcode = check_call(cmd+x_flag+f_flag+i_flag+o_flag)
    return eswc_path

def distance_transform(img_path):
    cmd = [VAA3D]
    x_flag = ['-x','gsdt']
    f_flag = ['-f', 'gsdt']
    i_flag = ['-i', img_path]
    p_flag = ['-p', '.5', '1', '0', '1']
    print img_path
    img_name, img_ext = os.path.splitext(img_path)
    tif_out = img_name + '-distancetransformed' + img_ext
    o_flag = ['-o', tif_out]
    retcode = check_call(cmd+x_flag+f_flag+i_flag+o_flag+p_flag)
    return tif_out

def gaussian_filter(img_path):
    cmd = [VAA3D]
    x_flag = ['-x', 'gaussian']
    f_flag = ['-f', 'gf']
    i_flag = ['-i', img_path]
    img_name, img_ext = os.path.splitext(img_path)
    tif_out = img_name + '-gf55512' + img_ext
    o_flag = ['-o', tif_out]
    p_flag = ['-p', '5', '5', '5', '1', '2']
    print img_path
    print tif_out
    print cmd+x_flag+f_flag+i_flag+o_flag+p_flag
    retcode = check_call(cmd+x_flag+f_flag+i_flag+o_flag+p_flag) 
    return tif_out

# input is already gf'd and (maybe) contrast enhanced
def trace_vasc(mpp, imgpath=None, img=None, gf=False, tmpdir=True, 
               retuse=False):
    base_name = str(time.time())
    # if tmpdir:
    #     tmpdir = tf.gettempdir()
    #     orig_dir = os.getcwd()
    #     os.chdir(tmpdir)
    if imgpath is None:
        img_base_name = base_name+'_base.tif'
        tiff.imsave(img_base_name, img)
    elif imgpath is not None:
        img_base_name = imgpath
    imshape = img.shape
    if gf:
        img_base_name = gaussian_filter(img_base_name)
    first_swc = snake_trace(img_base_name)
    first_swc = radius_fill(first_swc, img_base_name)
    first_swc_mask = swc_to_mask(first_swc, imshape)
    distance_mask = distance_transform(first_swc_mask)
    second_swc = snake_trace(distance_mask)
    second_swc = radius_fill(second_swc, img_base_name)
    second_swc_mask_segid = swc_to_mask(second_swc, imshape, segid=True)
    
    if tmpdir:
        os.chdir(orig_dir)
    if retuse:
        finished_swc = swc.SWC(path=second_swc, mpp=mpp)
        ret = (finished_swc, tiff.imread(second_swc_mask_segid))
    else:
        ret = (second_swc, second_swc_mask_segid)
    
def trace_all(paths, store=None):
    pairs = {}
    for path in paths:
        outs = trace_vasc((1, 1, 1), imgpath=path)
        pairs[path] = outs
    if store is not None:
        p.dump(pairs, open(store, 'wb'))
    return pairs
