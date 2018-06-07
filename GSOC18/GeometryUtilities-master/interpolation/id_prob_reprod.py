import uproot
import pandas as pd
import concurrent.futures,multiprocessing
ncpu=multiprocessing.cpu_count()
executor=concurrent.futures.ThreadPoolExecutor(ncpu*4)

from geometry.cmssw import read_geometry

import sys
import datetime
from scipy.spatial import cKDTree

############# Geometry File Helper Function ###################
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

############# Hit HGTuple File Helper Function ################
def readDataFile_hits(filename,event_id,layer_num):
    tree=uproot.open(filename)['ana/hgc']
    branches=[]
    branches += ["rechit_energy","rechit_detid",
                "rechit_x","rechit_y","rechit_z","rechit_layer"]
    cache={}
    df=tree.pandas.df(branches,cache=cache,executor=executor)

    #Projecting the dataframe for the required attributes
    print '>>> Selecting a single event'
    all_hits = pd.DataFrame({name.replace('rechit_',''):df.loc[event_id,name]
                            for name in branches if 'rechit_' in name })

    all_hits=all_hits[all_hits['layer']==layer_num]
    print all_hits.head()
    print all_hits.shape
    return all_hits

############# MAIN #########################
def test(df,hex_cells_dict,precision_adjust=1e-4):
    #getting the hit information
    hit_centers=df[['x','y']].values
    hit_z=df['z'].values
    hit_det=df['detid'].values
    #print hit_center.shape,hit_det.shape
    hit_cell_id=hit_det & 0x3FFFF   #taking out yellow and green part

    #getting the hex_cell information form geometry file
    hex_cells_list=hex_cells_dict.values()
    hex_centers=[cell.center.coords[0] for cell in hex_cells_list]
    hex_tree=cKDTree(hex_centers,balanced_tree=True)

    #Now querying the tree with hit centers to get corresponding
    #hex cells
    indices=hex_tree.query_ball_point(hit_centers,r=precision_adjust)

    for i,hex_pos in enumerate(indices):
        #Checking the on-one mapping between hex and hit cells
        if not len(hex_pos)==1:
            print 'Multiple cells detected for same hit'
            sys.exit(1)
        hex_pos=hex_pos[0]

        #now matching the ids
        if hit_cell_id[i]==hex_cells_list[hex_pos].id:
            print 'MATCH !'
        else:
            print 'NO MATCH !'
        print 'hit_det: ',hit_det[i]
        print 'hit_id:%s hex_id:%s '%(hit_cell_id[i],hex_cells_list[hex_pos].id)
        print 'hit_z: ',hit_z[i]
        print 'hit_pos: ',hit_centers[i]
        print 'hex_cell_pos: ',hex_cells_list[hex_pos].center.coords[0]
        print 'hit_cell_pos: ',hex_cells_dict[hit_cell_id[i]].center.coords[0]
        print '\n'

if __name__=='__main__':
    hits_default_filename="detector_data/hgcalNtuple_electrons_15GeV_n100.root"
    input_default_file="geometry_data/test_triggergeom.root"

    #Setting up the command line option parser
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input_geometry', dest='input_file',
                help='Input geometry file', default=input_default_file)
    parser.add_option('--data_file',dest='data_file',
                help='Ground Truth and Recorded Hits',default=hits_default_filename)
    (opt, args) = parser.parse_args()


    ############### TESTING ################################
    #Selecting a particular layer and event for test
    event_id=1
    layer_num=1
    hit_dataframe=readDataFile_hits(opt.data_file,event_id,layer_num)

    #Reading the geometry file for this layer
    subdet,eff_layer=get_subdet(layer_num)
    hex_cells_dict=readGeometry(opt.input_file,eff_layer,subdet)

    test(hit_dataframe,hex_cells_dict)
