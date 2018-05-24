##########################IMPORTS########################
#For timing script
import datetime
#For file IO/data Handling
import os
import sys
import cPickle as pickle
import pandas as pd
#Linear Algebra library
import numpy as np
from scipy.spatial import  cKDTree
#Plotting Imports and configuration
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Polygon
from descartes.patch import PolygonPatch
#Importing custom classes and function
from sq_Cells import sq_Cells


#################Global Variables#######################
dtype=np.float64            #data type of any numpy array created
sq_cells_basepath='sq_cells_data/'
if not os.path.exists(sq_cells_basepath):
    os.makedirs(sq_cells_basepath)

#################Function Definition####################
def linear_interpolate_hex_to_square(hex_cells_dict,sq_cells_dict,edge_length):
    '''
    DESCRIPTION:
        This function will interpolate the energy deposit in hexagonal cells
    from the input file to a energy deposit in the equivalent square grid
    Here we will interpolate according the area of overlap of a cell with
    the cells of square grid.

    INPUT:
        hex_cells_dict  : the dictionary of input geometry read from root file
        sq_cells_dict   : the common square cell mesh for interpolation
        edge_length     : the edge length of the square cells
    OUTPUT:
        coef(unnormalized) : a dictionary which contains the coefficient of overlap
                           for each cells with corresponding sqare cell and
                           fraction stored as:
                            { hex_center :[((i,j),cf),((i,j),cf)....]
                            }
    '''

    #Calculating the maximum length of any cells
        #(will to used to specify search radius in KD tree)
    print '>>> Calculating the Search Radius'
    t0=datetime.datetime.now()
    max_length_hex=max(map(
                    lambda c: max([
                    c.vertices.bounds[2]-c.vertices.bounds[0],
                    c.vertices.bounds[3]-c.vertices.bounds[1]
                    ]),hex_cells_dict.values())
                    )
    #DISCUSS and CONFIRM THIS LINE
    max_length_sq=np.sqrt(edge_length**2+edge_length**2 )
    #Any overlapping cells will be in this search radius
    search_radius=(max_length_hex/2)+(max_length_sq/2)
    t1=datetime.datetime.now()
    print 'Search Radius finding completed in: ',t1-t0,' sec'

    #Calculating the coefficient of overlap
    print '>>> Calculating the Overlap Coefficient'
    coef_dict=calculate_overlap(hex_cells_dict.values(),sq_cells_dict.values(),
                            search_radius,min_overlap_area=0.0)
    t2=datetime.datetime.now()
    print 'Overlap Coef Finding completed in: ',t2-t1,' sec'

    #Returning the coef_dict,resolution and the edge length
    return coef_dict

def generate_mesh(hex_cells_dict,edge_length,save_sq_cells=False):
    '''
    DESCRIPTION:
            This function will calculate the resolution based on the expected
        edge length. Then generate a common square mesh to be used by all the
        layers. Hence the layer with maximum dimension is taken to generate the
        mesh grid.
    USAGE:
        INPUT:
            hex_cells_dict  : the dictionary containing the hexagonal cells
            edge_length     : the edge length of each square cells
        OUTPUT:
            resolution      : the resolution of mesh grid for the given detector
                                layers at the current edge_length
            sq_cells_dict   : the square mesh dictionary for furthur interpolation
    '''
    #Iterating over all the cells to get the bounds of the detector
    print '>>> Calculating Bounds'
    t1=datetime.datetime.now()
    cell_bounds=map(lambda c:c.vertices.bounds,hex_cells_dict.values())
    max_x=max(bound[2] for bound in cell_bounds)
    min_x=min(bound[0] for bound in cell_bounds)
    max_y=max(bound[3] for bound in cell_bounds)
    min_y=min(bound[1] for bound in cell_bounds)
    t2=datetime.datetime.now()
    layer_bounds=(min_x,min_y,max_x,max_y)
    print 'Bounds: xmin:%s ,xmax:%s '%(min_x,max_x)
    print 'Bounds: ymin:%s ,ymax:%s '%(min_y,max_y)
    print 'Bounding completed in: ',t2-t1,' sec'

    #Padding the bounds to accomodate the cell with required edgelength
    print '>>> Calculating Resolution for the Meash Grid'
    pad_x=((max_x-min_x)%edge_length)/2
    pad_y=((max_y-min_y)%edge_length)/2
    #Padding with the offset (balancing both side)
    max_x=max_x+pad_x
    min_x=min_x-pad_x
    max_y=max_y+pad_y
    min_y=min_y-pad_y

    #Calculating the Resolution (based on edge_length)
    res_x=int(np.ceil((max_x-min_x)/edge_length))+1
    res_y=int(np.ceil((max_y-min_y)/edge_length))+1
    resolution=(res_x,res_y)
    print 'Calculated Resolution for edge_length:%s is: (%s %s)'%(edge_length,
                                                                res_x,res_y)

    #Creating the Square Mesh Grid
    print '>>> Generating the Square Mesh'
    sq_cells_dict=_get_square_cells(resolution,layer_bounds,edge_length,save_sq_cells)
    t3=datetime.datetime.now()
    print 'Generation Complete in: ',t3-t2,' sec'
    return resolution,sq_cells_dict

def _get_square_cells(resolution,layer_bounds,edge_length,save_sq_cells):
    '''
    DESCRIPTION:
        This function will generate square mesh grid by Creating
    the square polygon. This function create the square cell using the
    class defined in sq_Cells.py script. This first calculate the appropriate
    length of the square cells based on the resolution i.e total number of
    cells in each direction and distance available.
    USAGE:
        INPUT:
            resolution      : the number of cells in both x and y direction in
                                form of tuple (res_x,res_y)
            layer_bounds    : (min_x,min_y,max_x,max_y) of the layer geometry
            edge_length     : the edge length of the sq cell in the grid
            save_sq_cells   : a boolean whether to save the square cell geometry
                                default: False
        OUPUT:
            sq_cells        : a dictionary with the key as id of cell and
                                value as the square cell object.
                                {
                                key:(i,j) : value:(sqCell object)
                                }
            This dictionary is saved as pickle file in a new directory created
            automatically in current directory named as 'sq_cells_data'
    '''
    #Creating empty array to store
    #sq_cells=np.empty(resolution,dtype=np.object)
    sq_cells={}

    min_x,min_y,max_x,max_y=layer_bounds
    #Length in each dimension (Square box)
    x_length=edge_length
    y_length=edge_length

    #Time Comlexity = O(res[0]*res[1])
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            #Center of the square polygon
            #Now they wont coincide with actual center of polygon
            center=(min_x+i*x_length,min_y+j*y_length)
            id=(i,j)    #given in usual matrix notation
            sq_cells[id]=sq_Cells(id,center,x_length,y_length)

    #Saving the sq_cell sq_cell_data in given folder (Optional)
    if save_sq_cells==True:
        sq_cells_filename=sq_cells_basepath+'sq_cells_dict_res_%s,%s_len_%s.pkl'%(
                                    resolution[0],resolution[1],edge_length)
        fhandle=open(sq_cells_filename,'wb')
        pickle.dump(sq_cells,fhandle,protocol=pickle.HIGHEST_PROTOCOL)
        fhandle.close()

    return sq_cells

def calculate_overlap(hex_cells_list,sq_cells_list,search_radius,min_overlap_area=0.0):
    '''
    DESCRIPTION:
        This function calculate the overlap coeffieicnt from between the
        hexagonal cell and corresponding square cells. It is called internally
        by above linear_interpolate_hex_to_square frunciton.

        Generated a coefficient dictionary of form:
        { hexagon id 1: [(overlap_sq_cell_id,overlap_coefficient),(....),(.....)]
        }
    INPUT:
        hex_cells_list  : hexagonal cells in form of list
        sq_cells_list   : square cells in form of list
        search_radius   : upto what distance to search in KD-Tree.
        min_overlap_area: the minimum overlap with square cell to accept it
                            as candidate of overlap_cells
                            (default greater than 0.0)
    OUTPUT:
        coef_dict       : the dictionary containg the mapping of hexagonal cells
                            with the overlapping square cells and their
                            coefficient of overlap in form of fraction of area.
    '''
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
        #Storing the overlap area directly. Normalize later when using O(1)
        overlap_coef_final=overlap_area_final#/np.sum(overlap_area_final)

        #We are using the hex_center's center tuple as the key
        coef_dict[hex_cell.center.coords[0]]=[]
        #coef_dict[hex_cell.id]=[]
        for fid,coef in zip(sq_cell_id_final,overlap_coef_final):
            coef_dict[hex_cell.center.coords[0]].append((sq_cells_list[fid].id,coef))
            #coef_dict[hex_cell.id].append((sq_cells_list[fid].id,coef))

    return coef_dict

def plot_sq_cells(sq_cells_dict):
    '''
    DESCRIPTION:
        This function is to visualize the correctness of the
        generated square grid from the Square Ploygon generated by
        get_square_cells function above.
    USAGE:
        INPUT:
            cell_d  : this takes in the square cell dictionary generated
                        by the get_square_cells function above.
        OUTPUT:
            No outputs currently

    '''
    t0=datetime.datetime.now()
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    for id,cell in sq_cells_dict.items():
        poly=cell.polygon
        patch=PolygonPatch(poly,alpha=0.5,zorder=2,edgecolor='blue')
        ax1.add_patch(patch)
    t1=datetime.datetime.now()
    print '>>> Plot Completed in: ',t1-t0,' sec'
    ax1.set_xlim(-160, 160)
    ax1.set_ylim(-160, 160)
    ax1.set_aspect(1)
    plt.show()

def plot_hex_to_square_map(coef,hex_cells_dict,sq_cells_dict):
    '''
    DESCRIPTION:
        This function is for visualization of mapping of Hexagonal cells
        to the square cells, for checking the correctness of resolution
        of interpolation with the criteria as mentioned by Florian Sir,
        (one square cell not overlapping with more than three hexagon cells)
    USAGE:
        This function is called internally in main.py generate_interpolation
        function.
        INPUT:
            coef : the coef dictionary mapping each hexagonal cells to their
                    correspoinding square cells
            hex_cells_dict  : the dictionary of hexagonal cells obtained from
                                the root file
            sq_cells_dict   : the square cell dictionary generated by
                                get_square_cells function above and saved in
                                'sq_cells_data' directory in current location
        OUTPUT:
            Currently no output from this function
    '''
    t0=datetime.datetime.now()
    print '>>> Calculating the area of smallar cell for filtering'
    filter_hex_cells=([c.vertices.area for c in hex_cells_dict.values()
                        if len(list(c.vertices.exterior.coords))==7])
    small_wafer_area=min(filter_hex_cells)
    t1=datetime.datetime.now()
    print '>>> Area calculated %s in time: %s sec'%(
                                    small_wafer_area,t1-t0)
    t0=t1

    for hex_id,sq_overlaps in coef.items():
        hex_cell=hex_cells_dict[hex_id]
        poly=hex_cell.vertices
        #Filtering the cells in smaller region
        if poly.area!=small_wafer_area:
            continue

        fig=plt.figure()
        ax1=fig.add_subplot(111)
        x,y=poly.exterior.xy
        ax1.plot(x,y,'o',zorder=1)
        patch=PolygonPatch(poly,alpha=0.5,zorder=2,edgecolor='blue')
        ax1.add_patch(patch)
        print '>>> Plotting hex cell: ',hex_id
        for sq_cell_data in sq_overlaps:
            sq_cell_id=sq_cell_data[0]
            overlap_coef=sq_cell_data[1]
            sq_cell=sq_cells_dict[sq_cell_id]
            print ('overlapping with sq_cell: ',sq_cell_id,
                                    'with overlap coef: ',overlap_coef)
            poly=sq_cell.polygon
            x,y=poly.exterior.xy
            ax1.plot(x,y,'o',zorder=1)
            patch=PolygonPatch(poly,alpha=0.5,zorder=2,edgecolor='red')
            ax1.add_patch(patch)
        t1=datetime.datetime.now()
        print 'one hex cell overlap complete in: ',t1-t0,' sec\n'
        plt.show()

################'IMAGE' CREATION FUNCTION###############

def compute_energy_map(hex_cells_dict,coef,resolution,event_dataframe,
                        event_id,layer,precision_adjust=1e-5):
    '''
    DESCRIPTION:
        This function will finally map the energy deposit recorded in the
        hexagonal cell to the corresponding mapped square cells
        proportional to the coefficient of overlap calculated earlier.

        This function will process the whole event at once and generate
        as 3D map of the  energy deposit in the detector for that particular
        event.The output will then serve as an image to the CNN for futhur
        learning energy->particles mapping.
    CODE COMPLEXITY:
        O(number of hits*log(number of hex cells))
    USAGE:
        INPUT:
            hex_cells_dict  : the dictionary of hexagonal cells from the
                                input detector geometry.
            coef            : the mapping coefficient for cells in the required
                                layer specified by layer number
            resolution      : the square grid resolution for generating "image"
            event_dataframe : the pandas event dataframe read from root file
                                using the uproot library
            event_id        : could be a list of events for which to interpolate
                                (currently will support one event)
            layer           : the layer of which we are mapping the cells
            precision_adjust: to take into account that the data file of hgcal
                                hits are haveing rounded/low precision
                                position values. So searching the exact point
                                will not be possible.
        OUTPUT:
            energy_map      : a numpy array containing the map/interpolation
                                of a particular layer of a particular event.
                                (we should implement it for all the layers here
                                itself.and may be for all the event here later)
    '''
    #Projecting the dataframe for the required attributes
    print '>>> Projecting required attributes'
    rechits_attributes=["rechit_x", "rechit_y", "rechit_z",
                    "rechit_energy","rechit_layer", 'rechit_flags']
    project_dict={name.replace('rechit_',''):
                event_dataframe.loc[event_id,name] for name in rechits_attributes}
    all_hits=pd.DataFrame(project_dict)

    #Selection of rows which belong to the required layer
    print '>>> Selecting the rows belonging to layer: %s'%(layer)
    all_hits_layer=all_hits[all_hits['layer']==layer]
    print all_hits_layer.head()
    print all_hits_layer.dtypes

    #Getting the center of cells which have hots in that layer
    center_arr=all_hits_layer[['x','y']].values
    energy_arr=all_hits_layer['energy'].values
    print 'The datatype of center array is %s'%(center_arr.dtype)

    #Making the id center of each cell as tuple
    print '>>> Tuplizing the center of hits to make KD-Tree'
    hit_centers=[(center_arr[i,0],center_arr[i,1])
                    for i in range(center_arr.shape[0])]

    print '>>> Tuplizing the hexagonal cells center for efficient search'
    #order matters here since our result of query will be indices
    hex_cells_list=hex_cells_dict.values()
    hex_centers=[np.float32(cell.center.coords[0]).tolist()
                        for cell in hex_cells_list]
    hex_tree=cKDTree(hex_centers,balanced_tree=True)

    #Now querying the tree to get the corresponding cell to the
    #hit, search for the exact same point so r/distance=0
    #This will speed up the searching from N^2 to NlogN
    print '>>> querying the hex_cell_center_tree with the hit_centers'
    #O(hits*log(#hex_cells))
    indices=hex_tree.query_ball_point(hit_centers,r=precision_adjust)
    #print indices

    #FINALLY INTERPOLATING!!
    print '>>> Interpolating the hit Energy finally to square mesh'
    #Initializing the energy map array
    energy_map=np.zeros(resolution,dtype=dtype)
    for hit_id in range(energy_arr.shape[0]): #Complexity: O(#hits)
        hex_cell_index=indices[hit_id]
        if not len(hex_cell_index)==1:
            print 'Multiple/No Hex Cell Matching with hit cell'
            sys.exit(1)
            continue
        hex_cell=hex_cells_list[hex_cell_index[0]]
        #print np.float32(hex_cell.center.coords[0]).tolist(),hit_centers[hit_id]
        overlaps=coef[hex_cell.id]  #O(1) retreival
        for overlap in overlaps:    #O(1) constant due to our choice of grid res
            i,j=overlap[0]  #getting the sq_cell indices/id
            #now distributing the energy according to fraction of overlap
            energy_map[i,j]+=overlap[1]*energy_arr[hit_id]

    plt.imshow(energy_map)
    plt.colorbar()
    plt.show()
    return energy_map
