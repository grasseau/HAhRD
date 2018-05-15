#! /usr/bin/env python
import numpy as np
from geometry.zoltan_split import module_grid
import cPickle as pickle


def check_consistency(neighbors):
    for cell,nearest_neighbors in neighbors.items():
        for neighbor in nearest_neighbors:
            if not neighbor in neighbors:
                raise StandardError('ERROR: Neighbor not in the list of cells')
            neighbor_neighbors = neighbors[neighbor]
            if not cell in neighbor_neighbors:
                raise StandardError('ERROR: Cell not in the list of neighbors of its own neighbor')

def distances(neighbors):
    # Produce Zoltan/Split trigger cells
    # 8" flat to flat distance is 164.9mm -> 190.41mm vertex to vertex 
    print '> Producing Zoltan/Split geometry'
    modules_in = module_grid(19.041, 192, grid_size=13, triggercell_size=2)
    cells_in = [cell for module in modules_in for cell in module]
    cells_in_dict = dict([(c.id, c) for c in cells_in])
    distances = []
    for cell,nearest_neighbors in neighbors.items():
        for neighbor in nearest_neighbors:
            pos1 = cells_in_dict[cell].center
            pos2 = cells_in_dict[neighbor].center
            distances.append(pos1.distance(pos2))
    print '> Mean distance =', np.mean(distances)
    print '> RMS distances =', np.std(distances)
    print '> Min distance =', np.min(distances)
    print '> Max distance =', np.max(distances)
    return distances

def count(neighbors):
    number = []
    for cell,nearest_neighbors in neighbors.items():
        number.append(len(nearest_neighbors))
    print '> Mean number =', np.mean(number)
    print '> RMS number =', np.std(number)
    print '> Min number =', np.min(number)
    print '> Max number =', np.max(number)
    return number



def main(neighbor_file):  
    neighbors = pickle.load(open(neighbor_file, 'rb'))
    print 'Checking consistency'
    check_consistency(neighbors)
    print 'Checking distances'
    dists = distances(neighbors)
    print 'Checking numbers'
    numbers = count(neighbors)




if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--neighbors', dest='neighbor_file', help='Neighbor pickle file', default='neighbors.pkl')
    (opt, args) = parser.parse_args()
    main(opt.neighbor_file)
