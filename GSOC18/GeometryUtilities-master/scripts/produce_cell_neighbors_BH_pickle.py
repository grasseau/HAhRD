#! /usr/bin/env python

import cPickle as pickle
from geometry.cmssw import read_bh_geometry, compute_id
from geometry.neighbors import closest_neighbors
from geometry.cell import Cell,merge
import numpy as np
import ROOT
from array import array


def triggercell_mapping():
    ietamin = 17
    ietamax = 100
    iphimin = 1
    iphimax = 360
    triggercellsize = 3
    modulephisize = 10
    moduleetasize = (ietamax-ietamin+1)/3/2
    nmodulephi = (iphimax-iphimin+1)/triggercellsize/modulephisize
    mapping  = {}
    for iphi in xrange(iphimin, iphimax+1):
        iphi_tc = (iphi-iphimin)/triggercellsize+1
        iphi_mod = (iphi_tc-1)/modulephisize+1
        iphi_mod_cell = (iphi_tc-1)%modulephisize+1
        for ieta in xrange(ietamin, ietamax+1):
            ieta_tc = (ieta-ietamin)/triggercellsize+1
            ieta_mod = (ieta_tc-1)/moduleetasize+1
            ieta_mod_cell = (ieta_tc-1)%moduleetasize+1
            module_id = iphi_mod + (ieta_mod-1)*nmodulephi
            tc_id = iphi_mod_cell + (ieta_mod_cell-1)*modulephisize
            if not (module_id,tc_id) in mapping:
                mapping[(module_id,tc_id)] = []
            mapping[(module_id,tc_id)].append((ieta,iphi))
    return mapping

def create_triggercells(cells_dict):
    mapping = triggercell_mapping()
    trigger_cells = []
    for tc,cells in mapping.items():
        triggercell = merge([cells_dict[ieta_iphi] for ieta_iphi in cells])
        trigger_cells.append(Cell(
            id=compute_id(tc[0],tc[1]),
            layer=1,
            subdet=5,
            zside=1,
            module=tc[0],
            cell=tc[1],
            center=triggercell.centroid,
            vertices=triggercell
            ))
    return trigger_cells



def main(input_file, output_file):
    treename = 'hgcaltriggergeomtester/TreeCellsBH'
    cells = read_bh_geometry(filename=input_file, treename=treename)
    cells_dict = dict([((c.ieta,c.iphi), c) for c in cells])
    triggercells = create_triggercells(cells_dict)
    # Find max cell size
    max_size = max(map(lambda c:max([c.vertices.bounds[2]-c.vertices.bounds[0],c.vertices.bounds[3]-c.vertices.bounds[1]]), triggercells))
    # Find neighbors of each cell
    print 'Finding nearest neighbors'
    neighbor_indices, neighbor_ids = closest_neighbors(triggercells, max_distance=max_size)
    # Save mapping
    pickle.dump(neighbor_ids, open(output_file, 'wb'))



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input_geometry', dest='input_file', help='Input geometry file')
    parser.add_option('--output', dest='output_file', help='Output pickle file', default='neighbors.pkl')
    (opt, args) = parser.parse_args()
    if not opt.input_file:
        parser.print_help()
        print 'Error: Missing input geometry file name'
        sys.exit(1)
    main(opt.input_file, opt.output_file)
