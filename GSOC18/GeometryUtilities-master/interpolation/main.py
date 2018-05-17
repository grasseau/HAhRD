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
def generate_interpolation(hex_cell_dict_root,dataframe,layer,exp_edge_length=0.7):
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
            hex_cell_dict_root : the hex cell dictionary read from the
                                    root file.
            dataframe          : the dataframe with recorded energy deposits
            layer              : the layer for which interpolation coefficient
                                    need to be calculated
            exp_edge_length    : expected edge length of the square cell, from
                                    which the resolution will be calculated which
                                    fits with the layer bounds.
                                    This length will be aprroximate and actual
                                    length will be around it to have resolution
                                    as whole number
        OUTPUT:
            energy_map          : not currently added
    '''
    base_path=''
    ## Generating the overlapping coefficient
    hex_cells_dict=hex_cell_dict_root
    sq_coef,resolution,act_edge_length=linear_interpolate_hex_to_square(
                                            hex_cells_dict,layer,exp_edge_length)
    print '>>> Acual Edge Length %s,Resoultion %s'%(act_edge_length,resolution)
    #Saving the generated coefficient as pickle file
    coef_filename=base_path+'sq_cells_data/coef_dict_layer_%s_res_%s.pkl'%(layer,
                                                            resolution[0])
    fhandle=open(coef_filename,'wb')
    pickle.dump(sq_coef,fhandle,protocol=pickle.HIGHEST_PROTOCOL)
    fhandle.close()
    #Reading the pickle file of saved coefficient
    print '>>> Reading the Overlap Coefficient File'
    fhandle=open(coef_filename,'rb')
    sq_coef=pickle.load(fhandle)
    fhandle.close()


    ## Plotting the sq cell for verification
    print '>>> Reading the Square Cells File'
    sq_filename=base_path+'sq_cells_data/sq_cells_dict_layer_%s_res_%s.pkl'%(layer,
                                                            resolution[0])
    fhandle=open(sq_filename,'rb')
    sq_cells_dict=pickle.load(fhandle)
    fhandle.close()

    #plot_sq_cells(sq_cells_dict)
    plot_hex_to_square_map(sq_coef,hex_cells_dict,sq_cells_dict)

    #Calculating the ENERGY DEPOSIT map in the square grid from recorded hits
    #present in the dataframe
    #event_id=1
    #compute_energy_map(hex_cells_dict,sq_coef,resolution,dataframe,event_id,layer)


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
    if not opt.data_file:
        parser.print_help()
        print 'Error: Missing input data file name'
        sys.exit(1)
    if not opt.layer:
        parser.print_help()
        print 'Error: Please specify the layer to do interpolation'
        sys.exit(1)

    #Generating the required files ans calling the driver function
    cells_d = readGeometry( opt.input_file, opt.layer, opt.subdet )
    data_df= readDataFile(opt.data_file)

    #Calling the driver function
    generate_interpolation(cells_d,data_df,opt.layer,exp_edge_length=0.7)
