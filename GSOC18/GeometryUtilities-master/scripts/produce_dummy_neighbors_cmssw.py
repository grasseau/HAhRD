#! /usr/bin/env python

import cPickle as pickle
from geometry.cmssw import cell_id, wafer_id


def main(neighbor_file, output_file):  
    neighbors = pickle.load(open(neighbor_file, 'rb'))
    module_tc_neighbors = {}
    for cell,nearest_neighbors in neighbors.items():
        nns = []
        tc = cell_id(cell)
        module = wafer_id(cell)
        for neighbor in nearest_neighbors:
            tc_neighbor = cell_id(neighbor)
            module_neighbor = wafer_id(neighbor)
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
