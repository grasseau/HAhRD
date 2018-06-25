from main import get_subdet
from geometry.cmssw import read_geometry
import cPickle as pickle
import datetime

gfname='geometry_data/test_triggergeom.root'


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
    cells_d = dict([(c.id, c.center.coords[0]) for c in cells])
    t1 = datetime.datetime.now()
    print 'Cells read: number=', len(cells), ', time=', t1-t0
    return cells_d


if __name__=='__main__':
    total_layers=40
    for layer in range(1,total_layers+1):
        subdet,eff_layer=get_subdet(layer)
        hex_cells_pos_dict=readGeometry(gfname,eff_layer,subdet)

        #Saving the position
        print '>>> pickling layer %s\n'%(layer)
        fname='hex_pos_data/%s.pkl'%(layer)
        fhandle=open(fname,'wb')
        pickle.dump(hex_cells_pos_dict,fhandle,protocol=pickle.HIGHEST_PROTOCOL)
        fhandle.close()
