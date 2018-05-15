##########################IMPORTS########################
#For timing script
import datetime
#For file IO/data Handling
from six.moves import cPickle as pickle
#import pandas as pd
#Linear Algebra library
import numpy as np
from scipy.spatial import  cKDTree
#Plotting Imports and configuration
import matplotlib.pyplot as plt
#Importing custom classes and function
from sq_Cells import sq_Cells


#################Global Variables#######################
dtype=np.float64            #data type of any numpy array created


#################Function Definition####################
def linear_interpolate_hex_to_square(filename,layer,resolution=(1000,1000)):
    '''
    This function will interpolate the energy deposit in hexagonal cells
    from the input file to a energy deposit in the equivalent square grid
    Here we will interpolate according the area of overlap of a cell with
    the cells of square grid.

    INPUT:
        filename    :the pickled filename containing the input cell dict
        resolution  :(int,int) the resolution of the square grid (TUPLE)
        layer       :(int) the layer id
    OUTPUT:
        coef[all_hexagon,2] : contains the coefficient of overlap for
                each cells with corresponding sqare cell and fraction
                stored as:
                coef[hexid,[(i,j,cf),(i,j,cf)....]]
    '''
    t0=datetime.datetime.now()
    print '>>> Reading the HexCell File'
    infile=open(filename,'rb')
    cells_dict=pickle.load(infile)
    t1=datetime.datetime.now()
    print 'Reading completed in: ',t1-t0,' sec\n'

    #Creating the empty energy map (initialized with zeros)
    #coef=np.zeros(resolution,dtype=dtype)

    #Iterating over all the cells to get the bounds of the detector
    print '>>> Calculating Bounds'
    center_x=map(lambda c:c.center.x,cells_dict.values())
    max_x=max(center_x)
    min_x=min(center_x)
    center_y=map(lambda c:c.center.y,cells_dict.values())
    max_y=max(center_y)
    min_y=min(center_y)
    t2=datetime.datetime.now()
    print 'Bounding completed in: ',t2-t1,' sec\n'

    #Calculating the maximum length of any cells
        #(will to used to specify search radius in KD tree)
    print '>>> Calculating the search radius'
    max_length_hex=max(map(
                    lambda c: max([
                    c.vertices.bounds[2]-c.vertices.bounds[0],
                    c.vertices.bounds[3]-c.vertices.bounds[1]
                    ]),cells_dict.values())
                    )
    max_length_sq=np.sqrt( ((max_x-min_x)/(resolution[0]-1))**2+
                           ((max_y-min_y)/(resolution[1]-1))**2 )
    #Any overlapping cells will be in this search radius
    search_radius=(max_length_hex/2)+(max_length_sq/2)
    t3=datetime.datetime.now()
    print 'Search Radius finding completed in: ',t3-t2,' sec\n'

    #Getting the square cells mesh (dict) for overlap calculation
    print '>>> Generating the square mesh grid'
    sq_cells_dict=get_square_cells(resolution,min_x,min_y,max_x,max_y)
    t4=datetime.datetime.now()
    print 'Generating Mesh Grid completed in: ',t4-t3,' sec\n'

    #Calculating the coefficient of overlap
    #(currently in for of ditionary)
    print '>>> Calculating the overlap coefficient'
    coef=calculate_overlap(cells_dict.values(),sq_cells_dict.values(),
                            search_radius,min_overlap_area=0.0)
    t5=datetime.datetime.now()
    print 'Overlap Coef Finding completed in: ',t5-t4,' sec\n'

    #Now change it if we want the overlap with sq cells
    #in form of array
    #print coef
    return coef



def get_square_cells(resolution,min_x,min_y,max_x,max_y):
    ''' This function will generate square mesh grid by Creating
    the square polygon

    '''
    #Finding the dimension of each cells
    x_length=(max_x-min_x)/(resolution[0]-1)
    y_length=(max_y-min_y)/(resolution[1]-1)

    #Creating empty array to store
    #sq_cells=np.empty(resolution,dtype=np.object)
    sq_cells={}

    #Time Comlexity = O(res[0]*res[1])
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            #Center of the square polygon
            center=(min_x+i*x_length,min_y+j*y_length)
            id=(i,j)    #given in usual matrix notation
            sq_cells[id]=sq_Cells(id,center,x_length,y_length)

    return sq_cells

def calculate_overlap(hex_cells_list,sq_cells_list,search_radius,min_overlap_area=0.0):
    hex_centers=np.array([cell.center.coords[0]
                            for cell in hex_cells_list])
    sq_centers=np.array([cell.center.coords[0]
                            for cell in sq_cells_list])

    hex_kd_tree=cKDTree(hex_centers)
    sq_kd_tree=cKDTree(sq_centers)

    #Calculating all the possible overlaps of each hex cells
    #with all the sq cells
    overlap_candidate_id=hex_kd_tree.query_ball_tree(
                                    sq_kd_tree,search_radius)
    coef_dict={}
    for i,sq_cell_id in enumerate(overlap_candidate_id):
        #Going one by one for each cell and seiving through
        #all its overlap
        hex_cell=hex_cells_list[i]
        overlap_candidates=[sq_cells_list[j] for j in sq_cell_id]
        overlap_area=[]
        for overlap_candidate in overlap_candidates:
            overlap=hex_cell.vertices.intersection(
                        overlap_candidate.polygon)
            overlap_area.append(overlap.area)

        #Filtering the ones accoding to minimum ovelap criteria
        #by default the zero overlap cells are discarded
        sq_cell_id=np.array(sq_cell_id)
        overlap_area=np.array(overlap_area)
        selected_indices=overlap_area>min_overlap_area

        #Final accumulation of selected cell and their overlap area
        sq_cell_id_final=sq_cell_id[selected_indices]
        overlap_area_final=overlap_area[selected_indices]
        overlap_coef_final=overlap_area_final/np.sum(overlap_area_final)

        coef_dict[hex_cell.id]=[]
        for fid,coef in zip(sq_cell_id_final,overlap_coef_final):
            coef_dict[hex_cell.id].append((sq_cells_list[fid].id,coef))

    return coef_dict
