import numpy as np
from scipy.spatial import cKDTree



def map_cells(input_cells, output_cells, max_distance=5, min_overlap=0.001):
    input_position_array = np.array([cell.center.coords[0] for cell in input_cells])
    output_position_array = np.array([cell.center.coords[0] for cell in output_cells])
    input_tree = cKDTree(input_position_array)
    output_tree = cKDTree(output_position_array)
    # retrieve lists of (output) neighbors around input cells
    print '>> Fetching neighbors for the', len(input_cells), 'input cells'
    neighbors = input_tree.query_ball_tree(output_tree, max_distance)
    overlaps = []
    print '>> Computing overlaps'
    for i,cells in enumerate(neighbors):
        input_cell = input_cells[i]
        if len(cells)==0:
            raise RuntimeError('Cannot match input cell to any output cell')
        intersection_candidates = [output_cells[j] for j in cells]
        areas = []
        for intersection_candidate in intersection_candidates:
            intersection = input_cell.vertices.intersection(intersection_candidate.vertices)
            areas.append(intersection.area)
        areas = np.array(areas)
        areas /= np.sum(areas)
        cells = np.array(cells)
        intersection_indices = areas>min_overlap
        intersection_cells = cells[intersection_indices]
        intersection_sum = np.sum(areas[intersection_indices])
        overlaps.append(zip(intersection_cells, areas[intersection_indices]/intersection_sum))
    overlaps_dict = {}
    for cell,overlap in zip(input_cells,overlaps):
        overlaps_dict[cell.id] = []
        for output_cell, area in overlap:
            overlaps_dict[cell.id].append((output_cells[output_cell].id, area))
    return overlaps, overlaps_dict


def check_sharing_consistency(list1, list2):
    dict1 = dict(list1)
    dict2 = dict(list2)
    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    if len(set1-set2)>0: return False
    if len(set2-set1)>0: return False
    for key in set1:
        if not np.isclose(dict1[key], dict2[key]):
            return False
    return True

def check_mappings_consistency(mapping1, mapping2):
    keys1 = set(mapping1.keys())
    keys2 = set(mapping2.keys())
    keys1_only = keys1 - keys2
    keys2_only = keys2 - keys1
    keys_common = keys1 & keys2
    keys_common_eq = set(k for k in keys_common if check_sharing_consistency(mapping1[k], mapping2[k]))
    keys_common_neq = keys_common - keys_common_eq
    if len(keys_common_neq)>0:
        print 'Common not equal'
        for key in keys_common_neq:
            print key
            print '  1:', mapping1[key]
            print '  2:', mapping2[key]
    return len(keys_common_neq)>0

def merge_mappings(mapping1, mapping2):
    mapping_merged = mapping1.copy()
    mapping_merged.update(mapping2)
    return mapping_merged
