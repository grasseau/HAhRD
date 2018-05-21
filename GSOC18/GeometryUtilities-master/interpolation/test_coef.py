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
#This need to be manually entered
resolution = (514,513)
edge_length=0.7

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
        #norm=np.sum([sq[1] for sq in sq_coef[cell_id]])
        for sq in sq_coef[ cell_id ]:
            squareArea = squareArea + sq[1]#*area/norm

    print "Layer  Area :", layerArea
    print "Square Area :", squareArea

    if ( abs( (layerArea - squareArea)/layerArea ) < 1.e-07 ):
        print "Surface test PASS"
    else:
        print "Surface test ERROR"

    return abs( (layerArea - squareArea)/layerArea )

# Map the cell coef to a regular grid
def mappingOnMatrix( cells_d, sq_coef, resolution ):

    # Spread the coeficient in the squared grid
    sCells = np.zeros((resolution[0], resolution[1]) )
    for cell_id in sq_coef.keys():
        #area = cells_d[ cell_id].vertices.area
        #norm=np.sum([sq[1] for sq in sq_coef[cell_id]])
        for sq in sq_coef[ cell_id ]:
            i,j =  sq[0]
            sCells[i][j] = sCells[i][j] + sq[1]#*area/norm

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

    layers=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]
    #layers=[29]
    #resolution=(514,513)

    errors=[]
    for layer in layers:
        # Read cells (hexagons)
        subdet=None
        eff_layer=layer
        if layer<29:
            subdet=3
        else:
            subdet=4
            eff_layer=layer-28

        print 'layer:%s ,subdet:%s,eff_layer:%s '%(layer,subdet,eff_layer)
        cells_d = readGeometry( input_default_file, eff_layer, subdet )
        # Read coef
        fname='sq_cells_data/coef_dict_layer_%s_res_%s,%s_len_%s.pkl'%(
                                layer,resolution[0],resolution[1],edge_length)
        fhandle=open(fname,'rb')
        sq_coef=pickle.load(fhandle)
        fhandle.close()
        error=compareAreas( cells_d, sq_coef )
        errors.append(error)

        #sCells = mappingOnMatrix( cells_d, sq_coef, resolution )
        #plotImage( sCells)

    fig=plt.figure()
    fig.suptitle('Surface Area Error for different layers')

    ax1=fig.add_subplot(122)
    ax1.hist(errors)
    ax1.set_xlabel('Relative Error in Surface area of Mesh and Hex')
    ax1.set_ylabel('Count')

    ax2=fig.add_subplot(121)
    ax2.plot(layers,errors,'o')
    ax2.set_xlabel('layer no')
    ax2.set_ylabel('Surface Area Error')

    plt.show()
