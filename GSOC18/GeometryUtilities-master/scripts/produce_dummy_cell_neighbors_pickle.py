#! /usr/bin/env python

import cPickle as pickle
from geometry.neighbors import closest_neighbors
from geometry.cmssw import read_geometry
import numpy as np


def main(input_file, output_file, subdet):
    layers = [1]
    if subdet==3: layers = [1,28] 
    elif subdet==4: layers = [1,12] 
    # Read CMSSW geometry
    print 'Reading CMSSW geometry'
    treename = 'hgcaltriggergeomtester/TreeCells'
    cells_dict = {}
    for layer in layers:
        print '> Layer', layer
        cs = read_geometry(filename=input_file, treename=treename, subdet=subdet, layer=layer, wafer=-1)
        for c in cs:
            if c.id not in cells_dict: cells_dict[c.id] = c
    cells = cells_dict.values()
    # Find max cell size
    max_size = max(map(lambda c:max([c.vertices.bounds[2]-c.vertices.bounds[0],c.vertices.bounds[3]-c.vertices.bounds[1]]), cells))
    print max_size
    # Find neighbors of each cell
    print 'Finding nearest neighbors'
    neighbor_indices, neighbor_ids = closest_neighbors(cells, max_distance=max_size)

    # Save mapping
    pickle.dump(neighbor_ids, open(output_file, 'wb'))



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input_geometry', dest='input_file', help='Input geometry file')
    parser.add_option('--output', dest='output_file', help='Output pickle file', default='mapping.pkl')
    parser.add_option('--subdet', dest='subdet', help='Subdet', type='int', default=3)
    (opt, args) = parser.parse_args()
    if not opt.input_file:
        parser.print_help()
        print 'Error: Missing input geometry file name'
        sys.exit(1)
    main(opt.input_file, opt.output_file, opt.subdet)
