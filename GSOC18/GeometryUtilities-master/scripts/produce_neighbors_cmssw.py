#! /usr/bin/env python

import cPickle as pickle
from geometry.zoltan_split import cell_id as zoltan_cell_id
from geometry.zoltan_split import third_id as zoltan_third_id
from geometry.zoltan_split import module_id as zoltan_module_id
from geometry.zoltan_split import compute_id as zoltan_compute_id


def main(neighbor_file, output_file):  
    neighbors = pickle.load(open(neighbor_file, 'rb'))
    module_tc_neighbors = {}
    for cell,nearest_neighbors in neighbors.items():
        nns = []
        tc = (zoltan_third_id(cell)<<4) + zoltan_cell_id(cell)
        module = zoltan_module_id(cell)
        for neighbor in nearest_neighbors:
            tc_neighbor = (zoltan_third_id(neighbor)<<4) + zoltan_cell_id(neighbor)
            module_neighbor = zoltan_module_id(neighbor)
            nns.append((module_neighbor, tc_neighbor))
        module_tc_neighbors[(module, tc)] = nns

    with open(output_file, 'w') as f:
        for cell, neighbors in sorted(module_tc_neighbors.items()):
            line = '({0},{1})'.format(cell[0],cell[1])
            for neighbor in neighbors:
                line += ' ({0},{1})'.format(neighbor[0],neighbor[1])
            print >>f, line



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--neighbors', dest='neighbor_file', help='Neighbor pickle file', default='neighbors.pkl')
    parser.add_option('--output', dest='output_file', help='Output mapping file', default='neighbor_mapping.txt')
    (opt, args) = parser.parse_args()
    main(opt.neighbor_file, opt.output_file)
