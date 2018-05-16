#! /usr/bin/env python

import cPickle as pickle
from geometry.mapper import merge_mappings
from geometry.cmssw import cell_id as cmssw_cell_id
from geometry.cmssw import wafer_id as cmssw_wafer_id
from geometry.zoltan_split import cell_id as zoltan_cell_id
from geometry.zoltan_split import third_id as zoltan_third_id
from geometry.zoltan_split import module_id as zoltan_module_id
from geometry.zoltan_split import compute_id as zoltan_compute_id
from geometry.panels import compute_id as compute_panel_id



def main(input_file, output_file):  
    mapping = pickle.load(open(input_file, 'rb'))
    with open(output_file, 'w') as f:
        for module, panel in sorted(mapping.items()):
            print >>f, module, compute_panel_id(panel[0], panel[1])



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input', dest='input_file', help='Input mapping file', default='panel_mapping.py')
    parser.add_option('--output', dest='output_file', help='Output txt file', default='panel_mapping.txt')
    (opt, args) = parser.parse_args()
    main(opt.input_file, opt.output_file)
