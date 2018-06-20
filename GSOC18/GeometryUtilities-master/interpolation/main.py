#Helper function from this script
from hexCells_to_squareCell_interpolation import *

#General Imports
import sys
import datetime
import cPickle as pickle
import matplotlib.pyplot as plt

#Geometry File imports
from geometry.cmssw import read_geometry
input_default_file = '/data_CMS/cms/grasseau/HAhRD/test_triggergeom.root'
data_default_file = 'detector_data/hgcalNtuple_electrons_15GeV_n100.root'

#Data File imports
import uproot
import pandas as pd

#Impoeting the multiprocessing libraries
import concurrent.futures,multiprocessing
from functools import partial
ncpu=multiprocessing.cpu_count()
executor=concurrent.futures.ThreadPoolExecutor(ncpu*4)

############## DRIVER FUNCTION DEFINITION#############
def generate_interpolation(geometry_fname,edge_length=0.7):
    '''
    AUTHOR: Abhinav Kumar
    DESCRIPTION:
        This function is the main control point of generation of
        hexgonal cell to square cell interpolation. It is heavily
        dependent on functions of hexCell_to_sqyuareCell_interpolation.py
        STEPS:
            1.It calls the linear interpolation function to generate the
            coefficient of interpolation as dictionary.
            2.Then it saves the coefficient in form of pickle file in a
            separate folder in same directory named as 'sq_cells_data'.
            3.It plots the hexagon to square maps for few of sampled
            Hexagon cells.
    USAGE:
        INPUT:
            geometry_fname     : geometry root file of the detector
            edge_length        : edge length of the square cell, from
                                    which the resolution will be calculated which
                                    fits with the layer bounds.
        OUTPUT:(optional)
            coef_dict_array    : an array of size 52 have the interpolation
                                    coef of each layer in form:
                                    [coef_layer1,coef_layer2......]
    '''
    no_layers=40                  #[28:EE + 12:FH + 12:BH]

    #Generating the Common Mesh Grid to be used for all the layers
    print '>>> Generating Common Mesh Grid for All Layers'
    t0=datetime.datetime.now()
    #Reading Input Geometry
    subdet,eff_layer=get_subdet(no_layers)
    hex_cells_dict=readGeometry(geometry_fname,eff_layer,subdet)
    #Generating the Mesh Grid
    resolution,sq_cells_dict=generate_mesh(hex_cells_dict,edge_length,save_sq_cells=True)
    t1=datetime.datetime.now()
    print 'Generation of Mesh Grid Completed in: ',t1-t0,' time\n'

    #Generating the Overlapping Coefficient
    print '>>> Generating Overlapping Coefficient'

    #Starting to make different process for interpolation of different layers
    talpha=datetime.datetime.now()
    layers=range(1,no_layers+1)
    with multiprocessing.Manager() as manager:
        print '>>> Creating Shared Sq_cells_dict'
        #Creating a shared dict of square cells among all the process
        shared_sq_cells_dict=manager.dict(sq_cells_dict)

        #Creating the process
        print '>>> Starting the multiprocessing with %s process at a time'%(ncpu-2)
        process_pool=multiprocessing.Pool(processes=ncpu-2)
        #Now doing Map-Reduce to simultaneously run the processes
        process_pool.map(partial(interpolate_layer,
                            geometry_fname,shared_sq_cells_dict,edge_length,
                            resolution),layers)

    tbeta=datetime.datetime.now()
    print '>>>>> TASK COMPLETED in: ',tbeta-talpha

def interpolate_layer(geometry_fname,sq_cells_dict,edge_length,resolution,layer):
    #Reading the geometry file
    subdet,eff_layer=get_subdet(layer)
    hex_cells_dict=readGeometry(geometry_fname,eff_layer,subdet)

    #Calculating the sq_coef (unnormalized)
    sq_coef_dict=linear_interpolate_hex_to_square(hex_cells_dict,
                                            sq_cells_dict,edge_length)
    print 'Done for Layer:%s'%(layer)

    #Visual Consistency Check
    # print 'Checking for Consistency:'
    # sq_filename='sq_cells_data/sq_cells_dict_res_%s,%s_len_%s.pkl'%(
    #                             resolution[0],resolution[1],edge_length)
    # fhandle=open(sq_filename,'rb')
    # sq_cells_dict=pickle.load(fhandle)
    # fhandle.close()
    # plot_hex_to_square_map(sq_coef_dict,hex_cells_dict,sq_cells_dict)

    #Saving the coef_dict_array as a pickle
    print '>>> Pickling the coef_dict'
    coef_filename='sq_cells_data/coef_dict_layer_%s_res_%s,%s_len_%s.pkl'%(
                                layer,resolution[0],resolution[1],edge_length)
    t0=datetime.datetime.now()
    fhandle=open(coef_filename,'wb')
    pickle.dump(sq_coef_dict,fhandle,protocol=pickle.HIGHEST_PROTOCOL)
    fhandle.close()
    t1=datetime.datetime.now()
    print 'Pickling completed in: ',t1-t0,' sec\n'

def generate_image(hits_data_filename,resolution=(514,513),edge_length=0.7):
    #ONGOING
    '''
    DESCRIPTION:
        This funtion will read the dataframe and generate the "image" for futher
        CNN pipeline.This will read the event file and generate the square Grid
        interpolation for each event layer by layer thus creating a 3D image
        per event and finally a 4D dataset for CNN input combining all the event
    USAGE:
        INPUTS:
            hits_data_filename  : the filename for the hits data to read event
            resolution          : the resolution of current interpolation scheme
            edge_length         : the edge length of the current interpolation
                                    scheme
        OUTPUTS:

    '''
    #Some of the geometry metadata (will be constant)
    no_layers=40
    #Specifying the size of minibatch
    event_stride=20 #seems optimal in terms of memory use.
    event_start_no=0 #for testing now

    #Converting the root file to a data frame
    all_event_hits=readDataFile_hits(hits_data_filename,event_start_no,
                                    event_stride)

    t0=datetime.datetime.now()
    compute_energy_map(all_event_hits,resolution,edge_length,
                        event_start_no,event_stride,no_layers)
    t1=datetime.datetime.now()
    print '>>> Image Creation Completed in: ',t1-t0


################ MAIN FUNCTION DEFINITION ###################
def readGeometry( input_file,  layer, subdet ):
    '''
    AUTHOR: Grasseau Gilles
    DESCRIPTION:
        This function reads the root file which contain the Geometry
    of the detector and create a dictionary of "Cell" object assiciated
    with every hexagonal cell in the detector.
    USAGE:
        INPUT:
            input_file  : the name of input geometry file (root file)
            Layer       : which layer's cell we are interested in
            Subdet      : which part of subdetector it is
                            (EE,...)
        OUTPUT:
            cells_d     : the hexagonal cell-dictionary with id of
                          cell as the key and Cell object as value
    '''
    t0 = datetime.datetime.now()
    treename = 'hgcaltriggergeomtester/TreeCells'
    cells = read_geometry(filename=input_file, treename=treename,
              subdet=subdet, layer=layer, wafer=-1)
    cells_d = dict([(c.id, c) for c in cells])
    t1 = datetime.datetime.now()
    print 'Cells read: number=', len(cells), ', time=', t1-t0
    return cells_d

def get_subdet(layer):
    '''
    DESCRIPTION:
        This function calculates the subdet number and the effective layer
        number since the layers of the detector into three special sub-
        detectors which unique subdet number but non-unique eff_layer number
        which the read geometry function takes as input.
    USAGE:
        INPUT:
            layer       : the actual layer number in the actual detector geometry
        OUTPUT:
            subdet      : the number given to subdetectors of the HGCal
            eff_layer   : since the layers rollback to 1 for each subdet
    '''
    subdet=None
    eff_layer=None                  # Effective layer number (for rollback to 0)
    if layer<29:
        subdet=3                    # ECAL (Electromagnetic Calorimeter)
        eff_layer=layer
    elif layer<(29+12):
        subdet=4                    # Front HCAL (Hadron Calorimeter)
        eff_layer=layer-28
    elif layer<(29+12+12):
        subdet=5                    # Back HCAL (Hadronic Cal, Scintillator
        eff_layer=layer-28-12

    print 'Subdet selected: %s for layer: %s eff_layer: %s'%(subdet,
                                                            layer,eff_layer)
    return subdet,eff_layer

def readDataFile_hits(filename,event_start_no,event_stride):
    '''
    DESCRIPTION:
        This function will read the root file which contains the simulated
        data of particles and the corresponding recorded hits in the detector.
        The recorded hits in detertor will be later used for energy interpolation
        to the square cells.
        This code is similar to starting code in repo.
    USAGE:
        INPUT:
            filename        : the name of root file
            event_start_no  : the starting event number from where we want to
                                process minibatch.
            event_stride    : the size of minibatch to process in one go,
                                (the time cost taken is less than the memory
                                on increasing the value)
            query_string: this will be used to filter out the events like
                            selecting the hits in EE part with certain energy etc.
        OUTPUT:
            df          : the pandas dataframe of the data in root file
                            with only the recorded hits to convert to image
                            of the required batch size
    '''
    print '>>> Reading the root File to get hits dataframe'
    tree=uproot.open(filename)['ana/hgc']
    branches=[]
    #Just extracting the required attributes to create image
    branches += ["rechit_detid","rechit_energy"]
    #Adding the branches for logical Error check (Optional)
    #branches +=["rechit_z","rechit_cluster2d","cluster2d_multicluster"]
    #branches +=["rechit_cluster2d","cluster2d_multicluster"]

    cache={}
    df=tree.pandas.df(branches,cache=cache,executor=executor)

    #Renaming the attribute in short form
    col_names={name:name.replace('rechit_','') for name in branches}
    df.rename(col_names,inplace=True,axis=1)

    #Extracting out the minibatch of event to process at a time
    df=df.iloc[event_start_no:event_start_no+event_stride]

    #Do the Filtering here only no need to do it each time for each event

    #Printing for sanity check
    #print df.head()
    print 'Shape of dataframe: ',df.shape
    # print all_event_hits.loc[0,'energy']
    # print type(all_event_hits.loc[0,'energy'])
    # print all_event_hits.loc[0,'energy'].shape

    return df

def readDataFile_genpart(filename,event_start_no,event_stride):
    '''
    DESCRIPTION:
        This function is similar to readDataFile_hits but this will
        read the genpart of the same events as read by the above
        function to generate the target label for the corresponding
        image files.
    USAGE:
        INPUT:
            filename        : the name of the root file containing the
                                events
            event_start_no  : starting event number in this file to
                                extract the events from. This Will
                                be controlled manually while generating the
                                data for training set.
            event_stride    : the number of events to be processed in the
                                in one go.(consider memory cost here than the
                                time cost.)
        OUTPUT:
            df              : returns the data frame containing the particles
                                whose properties we will need to predict from
                                the corresponding hit images of events
    '''
    #Reading the root file to a dataframe
    print '>>> Reading the rootfile to get genpart dataframe'
    branches =["genpart_energy","genpart_phi","genpart_eta",
                "genpart_gen","genpart_pid","genpart_reachedEE"]
    cache={}
    df=tree.pandas.df(branches,cache=cache,executor=executor)

    #Renaming the attributes in short form
    col_names={name:name.replace('genpart_','') for name in branches}
    df.rename(col_names,inplace=True,axis=1)

    #Extracting the dataframe for the required events
    df=df.iloc[event_start_no:event_start_no+event_stride]

    print '>>> Extraction completed with current shape: ',df.shape

    return df

if __name__=='__main__':
    import sys

    #Setting up the command line option parser
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input_geometry', dest='input_file',
                help='Input geometry file', default=input_default_file)
    parser.add_option('--layer', dest='layer',
                help='Layer to be mapped', type='int', default=1)
    parser.add_option('--subdet', dest='subdet',
                help='Subdet', type='int', default=3)
    parser.add_option('--data_file',dest='data_file',
                help='Ground Truth and Recorded Hits',default=data_default_file)
    (opt, args) = parser.parse_args()

    #Checking if the required options are given or not
    if not opt.input_file:
        parser.print_help()
        print 'Error: Missing input geometry file name'
        sys.exit(1)
    # if not opt.data_file:
    #     parser.print_help()
    #     print 'Error: Missing input data file name'
    #     sys.exit(1)

    #Calling the driver function
    #generate_interpolation(opt.input_file,edge_length=0.7)

    #Generating the image
    generate_image(opt.data_file)
