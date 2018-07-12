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
#Importing a required function from main file
#from main import get_subdet as _get_subdet
#Importing Tensorflow to save the tfRecords
import tensorflow as tf


#################Global Variables#######################
dtype=np.float64            #data type of any numpy array created
#For saving the interpolation coefficents and sq_cells data
sq_cells_basepath='sq_cells_data/'
if not os.path.exists(sq_cells_basepath):
    os.makedirs(sq_cells_basepath)
#For saving the "image" after interpolation of hits
image_basepath='image_data/'
if not os.path.exists(image_basepath):
    os.makedirs(image_basepath)

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

        #We are using the hex_cell is as the key instead of cell center
        #coef_dict[hex_cell.center.coords[0]]=[]
        coef_dict[hex_cell.id]=[]
        for fid,coef in zip(sq_cell_id_final,overlap_coef_final):
            #coef_dict[hex_cell.center.coords[0]].append((sq_cells_list[fid].id,coef))
            coef_dict[hex_cell.id].append((sq_cells_list[fid].id,coef))

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
def _readCoefFile(filename):
    '''
    DESCRIPTION:
        To read the coef file. Will be used internally in the compute energy map
    '''
    fhandle=open(filename,'rb')
    coef_dict=pickle.load(fhandle)
    fhandle.close()

    return coef_dict

def _get_layer_number_or_mask_from_detid(detid,mask_layer=None):
    '''
    DESCRIPTION:
        This function will enable us to extract out the layer id from the
        full detid of hit in the data frame.
    USAGE:
        INPUT:
            detid           : a numpy arry having the detid of the hits
            mask_layer      : an optional layer-number to be used when
                              we want to get the layer mask for this given
                              layer number instead of layers list
        OUPUT:
            layer_arr       : a numpy array of unique layers
    '''
    #Getting effective layer numebr and the subdet number from the detid
    layer_arr=(detid>>19)&0x1F
    subdet_arr=(detid>>25)&0x7

    #Now making the effective layer to the actual number
    #Fixing subdet 4 layers
    layer_arr=layer_arr+(subdet_arr==4)*28
    #Fixing the subdet 5 layers
    layer_arr=layer_arr+(subdet_arr==5)*40

    #Generating the subdet mask to filter hit of required subdet
    subdet_mask=(subdet_arr==3) | (subdet_arr==4) | (subdet_arr==5)
    assert (subdet_mask.shape==layer_arr.shape),'Dim mismatch with mask'

    if mask_layer==None:
        #Now masking the layers which for the requires subdet
        return layer_arr[subdet_mask]
    else:
        #creating the net mask based on subdet and layer
        layer_mask=subdet_mask & (layer_arr==mask_layer)
        return layer_mask

def _get_cellid_energy_array(all_event_hits,layer,zside,event):
    '''
    DESCRIPTION:
        This will create the energy array and cellid whithout the extra memory
        overhead of mask.
    '''
    #getting the cellid by masking detid's last 18 bits
    detid=all_event_hits.loc[event,'detid']
    cellid_arr=detid & 0x3FFFF

    layer_mask=_get_layer_number_or_mask_from_detid(detid,layer)
    zside_mask=((detid>>24) & 0x1)==zside
    mask=layer_mask & zside_mask    #Final mnet mask

    #Now masking both the energy arr and cellid arr to retreive data of this layer
    energy_arr=np.squeeze(all_event_hits.loc[event,'energy']).reshape((-1,))
    #(LC) required for logical error check
    # z_arr=np.squeeze(all_event_hits.loc[event,'z']).reshape((-1,))
    # cluster2d_arr=np.squeeze(all_event_hits.loc[event,'cluster2d']).reshape((-1,))
    # cluster3d_arr=np.squeeze(
    #     all_event_hits.loc[event,'cluster2d_multicluster'][cluster2d_arr]).reshape((-1,))

    #Masking the array for requred zside->layer->event
    cellid_arr=cellid_arr[mask]
    energy_arr=energy_arr[mask]
    #masking (LC)
    # z_arr=z_arr[mask]
    # cluster3d_arr=cluster3d_arr[mask]

    return cellid_arr,energy_arr#,cluster3d_arr,z_arr

def _get_hit_layers(all_event_hits,event_start_no,event_stride):
    '''
    DESCRIPTION:
        This function will collect the set of all the layers which have hits in
        to be interpolated events.
    INPUT:
        all_event_hits  : the dataframe which contains the hit data (minibatch)
        event_start_no  : the starting point of interpolation of event
        event_stride    : the size of the minibatch to create image of
    OUTPUT:
        layer_set:  the unique list of hayers which have hits in any of the events
    '''
    layers=np.array([],dtype=np.int64)
    for event in range(event_start_no,event_start_no+event_stride):
        detid=all_event_hits.loc[event,'detid']
        _layers=np.unique(_get_layer_number_or_mask_from_detid(detid))
        layers=np.append(layers,_layers)

    layers=np.unique(layers)
    return layers.tolist()

def _bytes_feature(value):
    '''
    DESCRIPTION:
        Inspired/copied from the usual way to byte to Tensorflow
        example feature.
        Dont use it unknowingly.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def compute_energy_map(all_event_hits,event_mask,resolution,edge_length,event_file_no,
                    event_start_no,event_stride,no_layers,dtype=np.float32):
    '''
    DESCRIPTION:
        This function will finally map the energy deposit recorded in the
        hexagonal cell to the corresponding mapped square cells
        proportional to the coefficient of overlap calculated earlier.
        Multiprocessing Functionality could be added later if speed
        remains an issue.

        This function will interpolate the a minibatch of events dataframe
        and will generate a dataset ready for CNN.
        The output will then serve as an image to the CNN for futhur
        learning energy->particles mapping.
    CODE COMPLEXITY:

    USAGE:
        INPUT:
            all_event_hits  : the dataframe containing all the rechits from
                                all the events (a certain minibatch of events).
            event_mask      : a mask whether to select an event or not based
                                on the decision in the label creation function
            resolution      : the current resolution of the interpolation mesh
            event_file_no   : the file number of event which was used to generate
                                this (will give unique name to dataset)
            event_start_no  : the starting point of event number to create
                                minibatches.
            event_stride    : the size of minibatch to generate
            no_layers       : the total number of layers to interpolate upto
                                (current default is 40 since that much coef is
                                available to us right now)
            dtype           : np.float32 is kept as default to save memory
                                of the model
        OUTPUT:
            energy_map      : a numpy array containing the map/interpolation
                                of a minibatch of event.
    '''
    #(LC)For logical ERROR check
    # energy_diff=[]          #global list for tracking the error in
    # bary_x_diff=[]          # energy and the barycenter properties
    # bary_y_diff=[]
    # bary_z_diff=[]
    # cluster_properties={}  #For holding the details of cluster and actual data

    #(LC)Loading the sq_cells dict for sq cell pos
    # sq_cells_filename='sq_cells_data/sq_cells_dict_res_%s,%s_len_%s.pkl'%(
    #                             resolution[0],resolution[1],edge_length)
    # sq_cells_dict=_readCoefFile(sq_cells_filename)


    #Strating the tfRecord Writer
    for zside in [0,1]:
        image_filename=image_basepath+\
                    'image_event_file_%s_start_%s_stride_%s_zside_%s.tfrecords'%(
                                event_file_no,event_start_no,event_stride,zside)
        compression_options=tf.python_io.TFRecordOptions(
                        tf.python_io.TFRecordCompressionType.ZLIB)

        with tf.python_io.TFRecordWriter(image_filename,
                        options=compression_options) as record_writer:
            #Initializing the numpy matrix to hold the interpolation
            energy_map=np.zeros((event_stride,resolution[0],resolution[1],
                                    no_layers),dtype=dtype)

            #Starting to interpolate layer by layer for all the events
            layers=range(1,no_layers+1)
            #Better iterate only those layers whch are there in hit atleast once (LATER)
            #layers=_get_hit_layers(all_event_hits,event_start_no,event_stride)
            for layer in layers:
                #Loading the interpolation coef for this layer
                print '\n>>> Reading the layer %s interpolation coefficient'%(layer)
                coef_filename='sq_cells_data/coef_dict_layer_%s_res_%s,%s_len_%s.pkl'%(
                                            layer,resolution[0],resolution[1],edge_length)
                coef_dict=_readCoefFile(coef_filename)

                #(LC)Reading the position filename
                # pos_fname='hex_pos_data/%s.pkl'%(layer)
                # hex_pos=_readCoefFile(pos_fname)

                #Now we will iterate the all the events
                events=range(event_start_no,event_start_no+event_stride)
                for event in events:
                    #Filtering the event based on the event_mask
                    if event_mask[event-event_start_no]=='False':
                        continue

                    print '>>> Interpolating for Event:%s zside:%s'%(event,zside)
                    #Retreiving the data for that event of this layer(saving memory also)
                    print '>>> Masking and retreiving the hit'
                    hit_cellid_arr,hit_energy_arr=_get_cellid_energy_array(
                                            all_event_hits,layer,zside,event)
                    #(LC)
                    # hit_cellid_arr,hit_energy_arr,hit_cluster3d_arr,hit_z_arr=_get_cellid_energy_array(
                    #                         all_event_hits,layer,zside,event)

                    #Checking if the event contains no hits in this layer
                    if hit_energy_arr.shape[0]==0:
                        print 'Empty Event: ',hit_energy_arr.shape
                        continue

                    #Now iterating over all the hits of this layer in this event
                    for hit_id in range(hit_energy_arr.shape[0]):
                        #Accquiring the hexagonal cell
                        hex_cell_id=hit_cellid_arr[hit_id]

                        #Retreiving the overlap coef from the dictionary
                        overlaps=coef_dict[hex_cell_id]

                        #Performing the interpolation
                        hit_energy=hit_energy_arr[hit_id]

                        #(LC)Adding the new key to multi-cluster properties
                        # cluster3d=hit_cluster3d_arr[hit_id]
                        # if (event,cluster3d) not in cluster_properties.keys():
                        #     init_list=np.array([0,0,0,0],dtype=np.float64)
                        #     mesh_list=np.array([0,0,0,0],dtype=np.float64)
                        #     key=(event,cluster3d)
                        #     cluster_properties[key]=[init_list,mesh_list]
                        # #(LC)Now adding the hexagonal contribution to initial properties
                        # hex_cell_center=hex_pos[hex_cell_id]
                        # hit_Wx=hex_cell_center[0]*hit_energy
                        # hit_Wy=hex_cell_center[1]*hit_energy
                        # hit_Wz=hit_z_arr[hit_id]*hit_energy
                        # init_list=[hit_energy,hit_Wx,hit_Wy,hit_Wz]
                        # key=(event,cluster3d)
                        # cluster_properties[key][0]+=init_list

                        norm_coef=np.sum([overlap[1] for overlap in overlaps])
                        for overlap in overlaps:
                            #Calculating the interpolated/mesh energy for each overlap
                            i,j=overlap[0]  #index of square cell
                            weight=overlap[1]/norm_coef
                            mesh_energy=hit_energy*weight

                            #(LC) For adding the mesh contribution to mesh properties
                            # sq_center=sq_cells_dict[(i,j)].center
                            # mesh_energy=hit_energy*weight
                            # mesh_Wx=mesh_energy*sq_center.coords[0][0]
                            # mesh_Wy=mesh_energy*sq_center.coords[0][1]
                            # mesh_Wz=mesh_energy*hit_z_arr[hit_id]
                            # mesh_list=[mesh_energy,mesh_Wx,mesh_Wy,mesh_Wz]
                            # key=(event,cluster3d)
                            # cluster_properties[key][1]+=mesh_list

                            example_idx=event-event_start_no
                            energy_map[example_idx,i,j,layer-1]+=mesh_energy

            #Now saving the energy calculated for the particular z-side of event
            #REMEMBER: we have to retreive in this format only. also check
            #in what format numpy stores matrix by using tobytes.
            #(row mojor or column major)
            for example_idx in range(event_stride):
                #Not saving the events which were not interpolated
                if event_mask[example_idx]=='False':
                    continue
                print 'Making example for: ',example_idx
                example=tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(energy_map[example_idx,:,:,:].tobytes()),
                        #Adding an event lable to check sequential access
                        'event': _int64_feature(example_idx+event_start_no)
                    }
                ))
                record_writer.write(example.SerializeToString())

            #Testing the numpy array
            #np.save(image_filename,energy_map)

    #(LC)Appending the properties to the final error list
    # for key,value in cluster_properties.iteritems():
    #     #Normalizing the barycenters
    #     value[0][1:]=value[0][1:]/value[0][0]   #init_properties
    #     value[1][1:]=value[1][1:]/value[1][0]   #mesh_properties
    #
    #     #Appending the difference to the list
    #     energy_diff.append(np.abs((value[0][0]-value[1][0])))
    #     bary_x_diff.append(np.abs((value[0][1]-value[1][1])))
    #     bary_y_diff.append(np.abs((value[0][2]-value[1][2])))
    #     bary_z_diff.append(np.abs((value[0][3]-value[1][3])))
    #(LC) Plotting the values
    # from test_coef_multicluster import plot_error_histogram
    # plot_error_histogram(energy_diff,bary_x_diff,bary_y_diff,bary_z_diff)

    # We are not returning anything currently, but saving the tf records directly
    #return energy_map

############### TARGET CRETION FUNCTION################
def _int64_feature(value):
    '''
    DESCRIPTION:
        This function creates the int64 data to the examples
        features. If we have bytes which is taken by serializing
        a numpy array to bytes then used the above defined _bytes_feature.

    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def compute_target_lable(genpart_df,resolution,edge_length,
                        event_file_no,event_start_no,event_stride):
    '''
    DESCRIPTION:
        This function will create the target label for the their
        corresponding energy map for training the CNN.
        The target label will currently consist the example protocol
        which will save different field in target label as
        name-value pair in that protocol. Appropriate restructuring
        like converting the categorical variables to one-hot will be done
        while parsing.

    USAGE:
        INPUT:
            genpart_df      : dataframe of the events containing the
                                genpart information of the initial state of
                                particle.
            resolution      : the current resolution of interpolation in use
                                to be used for target representation on a mesh
            edge_length     : length of the current sq_cells in the mesh
            event_file_no   : the number on the event file to uniquely name dataset
            event_start_no  : the starting number of event slice from the root
                                file to be used for the anming convention
            event_stride    : the event stride i.e the batch size of the this
                                datafile i.e the number of examples in this data
                                file.
        OUTPUT:
            target_labels   : this will be a file saved in the imag_data directory
                                with the filename  convention decided inside
            event_mask      : a mas showing which event to take while Creating
                                the image to have a sync between the image and
                                label dataset
    '''
    #Reading the sq_cells dict for finding the probable location of
    #particle to the square layer
    # sq_cells_filename='sq_cells_data/sq_cells_dict_res_%s,%s_len_%s.pkl'%(
    #                             resolution[0],resolution[1],edge_length)
    # sq_cells_dict=_readCoefFile(sq_cells_filename)
    # sq_cells_center=np.array([cell.center.coords[0]
    #                     for cell in sq_cells_dict.value()])
    # #Creating the sq_cells KD-Tree
    # sq_kd_tree=cKDTree(sq_cells_center)

    #Making a flag for the events which dont have any electron
    count=0
    event_mask=[]

    #Setting up the filename and compression options of the target tfrecords
    label_filename=image_basepath+'label_event_file_%s_start_%s_stride_%s.tfrecords'%(
                                    event_file_no,event_start_no,event_stride)
    compression_options=tf.python_io.TFRecordOptions(
                    tf.python_io.TFRecordCompressionType.ZLIB)

    #Starting the tfrecords
    with tf.python_io.TFRecordWriter(label_filename,
                        options=compression_options) as record_writer:
        events=range(event_start_no,event_start_no+event_stride)
        for event in events:
            print '>>> Creating the target label for event: {}'.format(event)
            #Creating the mask for filtering the particles (on current requirement)
            electron_id=11
            positron_id=-11
            particles_mask=np.logical_or(genpart_df.loc[event,"pid"]==electron_id,
                                    genpart_df.loc[event,"pid"]==positron_id)
            particles_mask &= (genpart_df.loc[event,"gen"]>=0)
            particles_mask &= (genpart_df.loc[event,"reachedEE"]>1)
            particles_mask &= ((genpart_df.loc[event,"energy"]/
                                np.cosh(genpart_df.loc[event,"eta"]))>5)
            particles_mask &= (genpart_df.loc[event,"eta"])>0

            #Now filtering the required features for the target label
            particles_energy=genpart_df.loc[event,"energy"][particles_mask]
            particles_phi=genpart_df.loc[event,"phi"][particles_mask]
            particles_eta=genpart_df.loc[event,"eta"][particles_mask]
            particles_pid=genpart_df.loc[event,"pid"][particles_mask]
            #Getting the barycenter location form the position array
            particles_posx=np.array(genpart_df.loc[event,"posx"],
                                dtype=object)[particles_mask]
            particles_posy=np.array(genpart_df.loc[event,"posy"],
                                dtype=object)[particles_mask]
            particles_posz=np.array(genpart_df.loc[event,"posz"],
                                dtype=object)[particles_mask]

            #Checking if the electron is filtered and we got just one
            print particles_pid,'\n'
            if particles_pid.shape!=(1,): #and particles_pid.shape!=(2,):
                count+=1
                print 'Multiple/No Electron Detected!!!'
                event_mask.append('False') #Dont take this event
                continue
            else:
                event_mask.append('True') #Take this event
            #assert particles_pid.shape==(1,),"Multiple Electrons are detected"

            #Creating the label vector as its easier to manipulate in numpy
            #format: [energy,bary_posx,bary_posy,bary_posz,pc1(electron),pc2]
            #Target Metadata
            target_len=6
            label=np.empty((target_len,),dtype=np.float32)
            barycenter_depth=10 #as recommended by Florian and Arthur Sir
            #Filling up the target label
            label[0]=particles_energy[0]
            label[1]=particles_posx[0][barycenter_depth-1]
            label[2]=particles_posy[0][barycenter_depth-1]
            label[3]=particles_posz[0][barycenter_depth-1]
            if particles_pid[0]==11:
                label[4]=1      #its electron
                label[5]=0
            else:
                label[4]=0
                label[5]=1      #it not electon(positron ask Florian Sir)

            #Creating the example protocol to write to tfrecords
            #Add the event number later for check of the correspondancce
            #between the events in the label and image
            example=tf.train.Example(features=tf.train.Features(
                    feature={
                        #Saving each features as the named dict with bytes
                        'label':_bytes_feature(label.tobytes()),
                        #extra label of event for seq access check
                        'event':_int64_feature(event)
                    }
                )
            )
            #Adding the example to the record writer
            record_writer.write(example.SerializeToString())

    #Seeing the fraction of events which dont have just one electron
    print 'Total number of events skipped: ',count

    #Returning the event mask for event selection in image_creation
    return event_mask,target_len

################## MERGING IMAGE & LABELS ####################
def _binary_parse_function_image(serialized_example_protocol):
    '''
    DESCRIPTION:
        This furnction will be similar to the the parser function
        used in io pipeline but general purpose in termos of
        parameter which are hard coded there which need to be changed
        based on the resolution and the target/label length.
    USAGE:
        serialized_example_protocol : the binary serialized data
                                        in example protocol

    DONT use it directly. This will  be called internally
    while decoding the tf records.
    '''
    features={
        'image':tf.FixedLenFeature((),tf.string),
        'event':tf.FixedLenFeature((),tf.int64)
    }
    parsed_feature=tf.parse_single_example(
                    serialized_example_protocol,features)

    #Just returning the byte string so no need to convert again
    image=parsed_feature['image']
    #Reading the event id for matching and then removing it from examples
    event_id=tf.cast(parsed_feature['event'],tf.int32)

    return image,event_id

def _binary_parse_function_label(serialized_example_protocol):
    '''
    DESCRIPTION:
        This function will read the example feature and return the
        label as a byte string as it was saved during the tfrecords
        creation. So just directly save it as a features without the
        tobytes conversion.
    USAGE:
        serialized_example_protocol : the example protocoled version of our
                                        target label.
    DONT use it directly. THis will be passed a function to the dataset.map
    function.
    '''
    features={
        'label': tf.FixedLenFeature((),tf.string),
        'event': tf.FixedLenFeature((),tf.int64)
    }
    parsed_feature=tf.parse_single_example(
                    serialized_example_protocol,features)

    #Returing the byte string of the label
    label=parsed_feature['label']
    #Reading the event id to match with the corresponding image
    event_id=tf.cast(parsed_feature['event'],tf.int32)

    return label,event_id


def merge_image_and_label(event_file_no,event_start_no,event_stride):
    '''
    DESCRIPTION:
        This function will merge the image and the label together
        to give a dataset with image and label stitched together.
        Also, we need not shard the data cuz the approximate size of
        1000 images dataset is ~100 Mb so these files would be good
        to act as different shard.

        (TO Do Later)
        Later we wll migrate to the distributed setting then we will have
        tf.data inbuilt shard strategy wrt to the answer at:

        1.https://stackoverflow.com/questions/
                    48768206/how-to-use-dataset-shard-in-tensorflow
        2.https://www.tensorflow.org/deploy/distributed
    USAGE:
        INPUT:
            event_file_no   : the event file number from which the data is
                                extracted.
            event_start_no  : the starting event number from which the data
                                was extracted from the current event.
            event_stride    : the total number of events processed in this file
                                (just giving the range from the start point,
                                some of the events will be left based on filter mask)
    '''
    t0=datetime.datetime.now()
    for zside in [0,1]:
        image_filename=image_basepath+\
                    'image_event_file_%s_start_%s_stride_%s_zside_%s.tfrecords'%(
                                event_file_no,event_start_no,event_stride,zside)
        label_filename=image_basepath+\
                    'label_event_file_%s_start_%s_stride_%s.tfrecords'%(
                                        event_file_no,event_start_no,event_stride)

        #Specifying the compression type
        comp_type='ZLIB'
        #Reading the image and label dataset dataset
        dataset_image=tf.data.TFRecordDataset(image_filename,
                            compression_type=comp_type)
        dataset_label=tf.data.TFRecordDataset(label_filename,
                            compression_type=comp_type)

        #Mapping the serialized eample protocol to get the required elements
        dataset_image=dataset_image.map(_binary_parse_function_image)
        dataset_label=dataset_label.map(_binary_parse_function_label)

        dataset_both=tf.data.Dataset.zip((dataset_image,dataset_label))

        #Making the one shot iterator
        one_shot_iterator=dataset_both.make_one_shot_iterator()

        #Now saving the merged examples on by on in tfrecords
        compression_options=tf.python_io.TFRecordOptions(
                        tf.python_io.TFRecordCompressionType.ZLIB)
        merged_record_filename=image_basepath+\
                    'event_file_%s_start_%s_stride_%s_zside_%s.tfrecords'%(
                        event_file_no,event_start_no,event_stride,zside)

        with tf.python_io.TFRecordWriter(merged_record_filename,
                            options=compression_options) as record_writer:

            with tf.Session() as sess:
                while True:
                    try:
                        t_alpha=datetime.datetime.now()
                        ((image,image_event_id),
                            (label,label_event_id))=sess.run(one_shot_iterator.get_next())
                        t_beta=datetime.datetime.now()
                        assert (image_event_id==label_event_id),'Event_id mismatch'

                        print 'Creating the example for event id:',image_event_id,\
                                    'read in: ',t_beta-t_alpha
                        example=tf.train.Example(features=tf.train.Features(
                                feature={
                                    #Saving the image as a feature in this protocol
                                    'image':_bytes_feature(image),
                                    #Saving label for that image
                                    'label':_bytes_feature(label)
                                }
                            )
                        )
                        #Adding the example to the record writer
                        record_writer.write(example.SerializeToString())
                    except tf.errors.OutOfRangeError:
                        print 'Completed the merging for zside ',zside
                        break

    t1=datetime.datetime.now()
    print 'Merging Completed',t1-t0
