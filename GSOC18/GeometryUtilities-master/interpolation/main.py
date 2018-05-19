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
import concurrent.futures,multiprocessing
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
    no_layers=28                  #[28:EE + 12:FH + 12:BH]

    #Generating the Common Mesh Grid to be used for all the layers
    print '>>> Generating Common Mesh Grid for All Layers'
    t0=datetime.datetime.now()
    #Reading Input Geometry
    subdet=get_subdet(no_layers)
    hex_cells_dict=readGeometry(geometry_fname,no_layers,subdet)
    #Generating the Mesh Grid
    resolution,sq_cells_dict=generate_mesh(hex_cells_dict,edge_length,save_sq_cells=False)
    t1=datetime.datetime.now()
    print 'Generation of Mesh Grid Completed in: ',t1-t0,' time\n'

    #Generating the Overlapping Coefficient
    print '>>> Generating Overlapping Coefficient'
    coef_dict_array=np.array((no_layers,),dtype=np.object)

    for layer in range(1,no_layers+1):

        #Reading the geometry file
        subdet=get_subdet(layer)
        hex_cells_dict=readGeometry(geometry_fname,layer,subdet)

        #Calculating the sq_coef (unnormalized)
        sq_coef_dict=linear_interpolate_hex_to_square(hex_cells_dict,
                                                sq_cells_dict,edge_length)

        #Saving the sq_coef for this layer in array
        coef_dict_array[layer-1]=sq_coef_dict

        #Visual Consistency Check
        # print 'Checking for Consistency:'
        # sq_filename='sq_cells_data/sq_cells_dict_res_%s,%s_len_%s.pkl'%(
        #                             resolution[0],resolution[1],edge_length)
        # fhandle=open(sq_filename,'rb')
        # sq_cells_dict=pickle.load(fhandle)
        # fhandle.close()
        # plot_hex_to_square_map(sq_coef_dict,hex_cells_dict,sq_cells_dict)

    #Saving the coef_dict_array as a pickle
    #(h5 formats are more memory efficient)
    print '>>> Pickling the coef_dict_array'
    coef_filename='sq_cells_data/coef_dict_res_%s,%s_len_%s.pkl'%(
                                    resolution[0],resolution[1],edge_length)
    t0=datetime.datetime.now()
    fhandle=open(coef_filename,'wb')
    pickle.dump(coef_dict_array,fhandle,protocol=pickle.HIGHEST_PROTOCOL)
    fhandle.close()
    t1=datetime.datetime.now()
    print 'Pickling completed in: ',t1-t0,' sec'

    return coef_dict_array

def generate_image(hits_data_fname,coef_dict_array_fname):
    '''
    DESCRIPTION:
        This funtion will read the dataframe and generate the "image" for futher
        CNN pipeline.This will read the event file and generate the square Grid
        interpolation for each event layer by layer thus creating a 3D image
        per event and finally a 4D dataset for CNN input combining all the event
    USAGE:
        INPUTS:
            hits_data_filename  : the filename for the hits data to read event
            coef_dict_array_fname: filename of saved interpolation Coefficient.
        OUTPUTS:

    '''
    #Some of the geometry metadata (will be constant)
    no_layers=52
    #Converting the root file to a data frame
    hits_df=readDataFile(hits_data_filename)

    #Initializing the numpy array to hold 4D data
    dataset=np.empty(())



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
    subdet=None
    if layer<29:
        subdet=3                    # ECAL (Electromagnetic Calorimeter)
    elif layer<(29+12):
        subdet=4                    # Front HCAL (Hadron Calorimeter)
    elif layer<(29+12+12):
        subdet=5                    # Back HCAL (Hadronic Cal, Scintillator

    print 'Subdet selected: %s for layer: %s'%(subdet,layer)
    return subdet

def readDataFile(filename):
    '''
    DESCRIPTION:
        This function will read the root file which contains the simulated
        data of particles and the corresponding recorded hits in the detector.
        The recorded hits in detertor will be later used for energy interpolation
        to the square cells.
        This code is similar to starting code in repo.
    USAGE:
        INPUT:
            filename    : the name of root file
        OUTPUT:
            df          : the pandas dataframe of the data in root file
    '''
    tree=uproot.open(filename)['ana/hgc']
    branches=[]
    branches += ["genpart_gen","genpart_reachedEE","genpart_energy",
                "genpart_eta","genpart_phi", "genpart_pid","genpart_posx",
                "genpart_posy","genpart_posz"]
    branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy",
                "rechit_layer", 'rechit_flags','rechit_cluster2d',
                'cluster2d_multicluster']
    cache={}
    df=tree.pandas.df(branches,cache=cache,executor=executor)

    #Testings
    # event_id=12
    # cluster=df.loc[event_id,'rechit_cluster2d']
    # mcluster=df.loc[event_id,'cluster2d_multicluster']
    # layer=df.loc[event_id,'rechit_layer']
    #
    # print cluster.shape,mcluster.shape,layer.shape
    # for i in range(layer.shape[0]):
    #     print layer[i],cluster[i]
    # for i in range(mcluster.shape[0]):
    #     print mcluster[i]

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
    # if not opt.layer:
    #     parser.print_help()
    #     print 'Error: Please specify the layer to do interpolation'
    #     sys.exit(1)

    #Calling the driver function
    #generate_interpolation(opt.input_file,edge_length=0.7)

    data_df= readDataFile(opt.data_file)
