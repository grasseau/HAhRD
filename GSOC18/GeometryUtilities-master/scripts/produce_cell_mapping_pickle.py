#! /usr/bin/env python

import datetime
import cPickle as pickle
from geometry.zoltan_split import module_grid
from geometry.cmssw import read_geometry
from geometry.mapper import map_cells
import numpy as np


def main(input_file, output_file, layer, subdet):
    # Read CMSSW geometry
    print 'Reading CMSSW geometry'
    t0 = datetime.datetime.now()
    treename = 'hgcaltriggergeomtester/TreeCells'
    cells = read_geometry(filename=input_file, treename=treename, subdet=subdet, layer=layer, wafer=-1)
    cells_dict = dict([(c.id, c) for c in cells])
    t1 = datetime.datetime.now()
    print '->', (t1-t0).seconds, 'seconds'
    # Produce Zoltan/Split trigger cells
    # 8" flat to flat distance is 164.9mm -> 190.41mm vertex to vertex 
    print 'Producing Zoltan/Split geometry'
    modules_out = module_grid(19.041, 192, grid_size=13, triggercell_size=2)
    cells_out = [cell for module in modules_out for cell in module]
    cells_out_dict = dict([(c.id, c) for c in cells])
    cells_out_module = dict([(cell.id,imod) for imod,module in enumerate(modules_out) for cell in module])
    # Find max output cell size
    max_size = max(map(lambda c:max([c.vertices.bounds[2]-c.vertices.bounds[0],c.vertices.bounds[3]-c.vertices.bounds[1]]), cells_out))
    t2 = datetime.datetime.now()
    print '->', (t2-t1).seconds, 'seconds'
    # Match CMSSW cells and Zoltan/Split trigger cells
    print 'Matching geometries'
    matched_indices, matched_ids = map_cells(cells, cells_out, max_distance=max_size)
    t3 = datetime.datetime.now()
    print '->', (t3-t2).seconds, 'seconds'

    # Save mapping
    pickle.dump(matched_ids, open(output_file, 'wb'))



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input_geometry', dest='input_file', help='Input geometry file')
    parser.add_option('--output', dest='output_file', help='Output pickle file', default='mapping.pkl')
    parser.add_option('--layer', dest='layer', help='Layer to be mapped', type='int', default=1)
    parser.add_option('--subdet', dest='subdet', help='Subdet', type='int', default=3)
    (opt, args) = parser.parse_args()
    if not opt.input_file:
        parser.print_help()
        print 'Error: Missing input geometry file name'
        sys.exit(1)
    main(opt.input_file, opt.output_file, opt.layer, opt.subdet)
