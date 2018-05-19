##########################IMPORTS########################
#For timing script
import datetime
#For file IO/data Handling
import os
import cPickle as pickle
#import pandas as pd
#Linear Algebra library
import numpy as np
from scipy.spatial import  cKDTree
#Plotting Imports and configuration
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Polygon
from descartes.patch import PolygonPatch
#Geometry File imports
from geometry.cmssw import read_geometry
#Importing custom classes and function
from sq_Cells import sq_Cells
from scipy import misc

# Parameters ... to fix later
#input_default_file = '/data_CMS/cms/grasseau/HAhRD/test_triggergeom.root'
input_default_file='geometry_data/test_triggergeom.root'
coef_filename='sq_cells_data/coef_dict_res_473,473_len_0.7.pkl'
#This need to be manually entered
resolution = (473,473)

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

# Compute and copare the total areas (hex, square)
def compareAreas( cells_d, sq_coef):
    layerArea = float(0.0)
    for k in cells_d.keys():
        layerArea =  layerArea + cells_d[k].vertices.area

    squareArea = float( 0.0 )
    for cell_id in sq_coef.keys():
        #area = cells_d[ cell_id ].vertices.area
        for sq in sq_coef[ cell_id ]:
            squareArea = squareArea + sq[1]#*area

    print "Layer  Area :", layerArea
    print "Square Area :", squareArea

    if ( abs( (layerArea - squareArea)/layerArea ) < 1.e-07 ):
        print "Surface test PASS"
    else:
        print "Surface test ERROR"

    return

# Map the cell coef to a regular grid
def mappingOnMatrix( cells_d, sq_coef, resolution ):

    # Spread the coeficient in the squared grid
    sCells = np.zeros((resolution[0], resolution[1]) )
    for cell_id in sq_coef.keys():
        #area = cells_d[ cell_id].vertices.area
        for sq in sq_coef[ cell_id ]:
            i,j =  sq[0]
            sCells[i][j] = sCells[i][j] + sq[1]#*area

    return sCells


# Plot the image ..
def plotImage( sCells ):

    iSize, jSize = sCells.shape
    sMax = sCells.max()
    print 'The maximum value is: ',sMax

    ima = np.zeros( (iSize, jSize, 3), dtype=np.uint8)
    for i in range(iSize):
        for j in range(jSize):
            val = int( sCells[i][j] / sMax * 255)
            if ((val >= 254) or (val == 0)):
                ima[i][j][0] = 255
            else:
                ima[i][j] = val
            # if sCells[i][j]==sMax:
            #     print i,j

    f = misc.face(gray=True)
    plt.imshow(f)
    plt.title("... !!! ...")
    plt.show()

    plt.imshow(sCells,  cmap=plt.cm.gray )
    plt.title("Raw image (no filter)")
    plt.show()

    plt.imshow(ima)
    plt.title("O and >249 values set to 255 (red)")
    plt.show()


if __name__=='__main__':

    # Read cells (hexagons)
    cells_d = readGeometry( input_default_file, 1, 3 )

    # Read coef
    fhandle=open(coef_filename,'rb')
    coef_dict_array=pickle.load(fhandle)
    fhandle.close()
    sq_coef=coef_dict_array[0]  #for layer 1

    compareAreas( cells_d, sq_coef )

    sCells = mappingOnMatrix( cells_d, sq_coef, resolution )

    plotImage( sCells)
