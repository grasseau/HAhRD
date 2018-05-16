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


############## DRIVER FUNCTION DEFINITION#############
def generate_interpolation(hex_cell_dict_root,resolution=(500,500)):
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
            resolution         : the resolution of square grid to be generated
                                    a tuple of form (res_x,res_y)
        OUTPUT:
            energy_map          : not currently added
    '''
    base_path=''
    ## Generating the overlapping coefficient
    hex_cells_dict=hex_cell_dict_root
    #resolution=(500,500)
    layer=1
    sq_coef=linear_interpolate_hex_to_square(hex_cells_dict,
                                                layer,resolution)
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


################ DRIVER FUNCTION DEFINITION ###################
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

if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input_geometry', dest='input_file', help='Input geometry file', default=input_default_file)
    # Not used
    # parser.add_option('--output', dest='output_file', help='Output pickle file', default='mapping.pkl')
    parser.add_option('--layer', dest='layer', help='Layer to be mapped', type='int', default=1)
    parser.add_option('--subdet', dest='subdet', help='Subdet', type='int', default=3)
    (opt, args) = parser.parse_args()
    if not opt.input_file:
      parser.print_help()
      print 'Error: Missing input geometry file name'
      sys.exit(1)
    cells_d = readGeometry( opt.input_file, opt.layer, opt.subdet )
    generate_interpolation(cells_d,resolution=(500,500))
