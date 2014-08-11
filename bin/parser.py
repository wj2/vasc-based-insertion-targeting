
import argparse as 

def damage_quant_parser():
    parser = ap.ArgumentParser(description='quantify theoretical damage from '
                               'electrode insertion')
    parser.add_argument('pre',help='path to 2-photon pre-data')
    parser.add_argument('post',help='path to 2-photon post-data')
    parser.add_argument('probe_length',help='length of probe used for '
                        'insertion')
    parser.add_argument('probe_width',help='width of probe used for insertion')

