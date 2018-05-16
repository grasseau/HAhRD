import numpy as np
from scipy.spatial import cKDTree


def closest_neighbors(input_cells, max_distance=5):
    input_position_array = np.array([cell.center.coords[0] for cell in input_cells])
    input_tree = cKDTree(input_position_array)
    print '>> Fetching neighbors for the', len(input_cells), 'input cells'
    neighbors = input_tree.query_ball_tree(input_tree, max_distance)
    nearest_neighbors = []
    print '>> Looking for nearest neighbors' 
    for i,cells in enumerate(neighbors):
        if i%(len(neighbors)/100)==0:
            print i, '/', len(neighbors)
        input_cell = input_cells[i]
        if len(cells)==0:
            raise RuntimeError('Cannot find any neighbor')
        intersection_candidates = [input_cells[j] for j in cells]
        areas = []
        for intersection_candidate in intersection_candidates:
            margin = min(input_cell.vertices.length, intersection_candidate.vertices.length)/1000.
            intersection = input_cell.vertices.intersection(intersection_candidate.vertices.buffer(margin))
            areas.append(intersection.area)
        areas = np.array(areas)
        cells = np.array(cells)
        intersection_indices = areas>0
        intersection_cells = cells[intersection_indices]
        nearest_neighbors.append(intersection_cells)
    neighbors_dict = {}
    for cell,nn in zip(input_cells,nearest_neighbors):
        neighbors_dict[cell.id] = []
        for neighbor in nn:
            if input_cells[neighbor].id!=cell.id:
                neighbors_dict[cell.id].append(input_cells[neighbor].id)
    return nearest_neighbors, neighbors_dict
