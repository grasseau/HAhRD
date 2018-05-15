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


def write_trigger_cell_mapping(input_template,  output_file):
    file_3_1 = '{}_3_1.pkl'.format(input_template)
    file_3_28 = '{}_3_28.pkl'.format(input_template)
    file_4_1 = '{}_4_1.pkl'.format(input_template)
    file_4_12 = '{}_4_12.pkl'.format(input_template)
    mapping_3_1 = pickle.load(open(file_3_1, 'rb'))
    mapping_3_28 = pickle.load(open(file_3_28, 'rb'))
    mapping_3 = merge_mappings(mapping_3_1, mapping_3_28)
    mapping_4_1 = pickle.load(open(file_4_1, 'rb'))
    mapping_4_12 = pickle.load(open(file_4_12, 'rb'))
    mapping_4 = merge_mappings(mapping_4_1, mapping_4_12)

    cells_to_trigger_cell = {}
    # Map cells to TC with maximum overlap
    # EE
    for cell,trigger_cells in mapping_3.items():
        matched_tc = max(trigger_cells, key=lambda x:x[1])[0]
        trigger_cell = (zoltan_third_id(matched_tc)<<4) + zoltan_cell_id(matched_tc)
        cells_to_trigger_cell[(3,cmssw_wafer_id(cell), cmssw_cell_id(cell))] = (zoltan_module_id(matched_tc), trigger_cell)
    # FH
    for cell,trigger_cells in mapping_4.items():
        matched_tc = max(trigger_cells, key=lambda x:x[1])[0]
        trigger_cell = (zoltan_third_id(matched_tc)<<4) + zoltan_cell_id(matched_tc)
        cells_to_trigger_cell[(4,cmssw_wafer_id(cell), cmssw_cell_id(cell))] = (zoltan_module_id(matched_tc), trigger_cell)

    with open(output_file, 'w') as f:
        for cell, tc in sorted(cells_to_trigger_cell.items()):
            print >>f, cell[0], cell[1], cell[2], tc[0], tc[1]

def write_panel_mapping(output_file):
    input_file = 'panel_mapping.pkl'
    mapping = pickle.load(open(input_file, 'rb'))
    with open(output_file, 'w') as f:
        for module, panel in sorted(mapping.items()):
            print >>f, module, compute_panel_id(panel[0], panel[1])

def main(input_template, output_trigger_cells, output_panels):  
    write_trigger_cell_mapping(input_template, output_trigger_cells)
    write_panel_mapping(output_panels)



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--inputs', dest='input_maps', help='Input mapping files template', default='mapping')
    parser.add_option('--tcfile', dest='output_trigger_cells', help='Output txt file for trigger cell mapping', default='mapping_tc.txt')
    parser.add_option('--panelfile', dest='output_panels', help='Output txt file for panel mapping', default='mapping_panel.txt')
    (opt, args) = parser.parse_args()
    main(opt.input_maps, opt.output_trigger_cells, opt.output_panels)
