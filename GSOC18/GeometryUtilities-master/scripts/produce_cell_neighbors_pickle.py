#! /usr/bin/env python

import cPickle as pickle
from geometry.zoltan_split import module_grid
from geometry.neighbors import closest_neighbors
import numpy as np


def main(output_file):
    # Produce Zoltan/Split trigger cells
    # 8" flat to flat distance is 164.9mm -> 190.41mm vertex to vertex 
    print 'Producing Zoltan/Split geometry'
    modules_in = module_grid(19.041, 192, grid_size=13, triggercell_size=2)
    cells_in = [cell for module in modules_in for cell in module]
    cells_in_dict = dict([(c.id, c) for c in cells_in])
    cells_in_module = dict([(cell.id,imod) for imod,module in enumerate(modules_in) for cell in module])
    # Find max cell size
    max_size = max(map(lambda c:max([c.vertices.bounds[2]-c.vertices.bounds[0],c.vertices.bounds[3]-c.vertices.bounds[1]]), cells_in))
    print max_size
    # Find neighbors of each cell
    print 'Finding nearest neighbors'
    neighbor_indices, neighbor_ids = closest_neighbors(cells_in, max_distance=max_size)

    # Save mapping
    pickle.dump(neighbor_ids, open(output_file, 'wb'))



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--output', dest='output_file', help='Output pickle file', default='neighbors.pkl')
    (opt, args) = parser.parse_args()
    main(opt.output_file)
