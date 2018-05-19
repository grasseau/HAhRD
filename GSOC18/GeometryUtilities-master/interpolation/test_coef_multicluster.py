import sys
import cPickle as pickle
from scipy.spatial import cKDTree

import uproot
import pandas as pd
import concurrent.futures,multiprocessing
ncpu=multiprocessing.cpu_count()
executor=concurrent.futures.ThreadPoolExecutor(ncpu*4)

#Location of the root data file
dfname='detector_data/hgcalNtuple_electrons_15GeV_n100.root'
cfname='sq_cells_data/coef_dict_res_473,473_len_0.7.pkl'

############# HELPER FUNCTION ###############
def readCoefFile(filename):
    fhandle=open(filename,'rb')
    coef_dict_array=pickle.load(fhandle)
    fhandle.close()

    return fhandle

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

    #Selecting hits from a particular event
    event_id=12
    all_hits = pd.DataFrame({name.replace('rechit_',''):df.loc[event_id,name]
                            for name in branches if 'rechit_' in name })

    #Generating the multicluster index
    cl2d_idx=df.loc[event_id,'rechit_cluster2d']
    mcl_idx=df.loc[event_id,'cluster2d_multicluster'][cl2d_idx]

    #Adding it to all hits data frame
    all_hits['cluster3d'] = pd.Series(mcl_idx, index=all_hits.index)
    max_mcl_idx=np.max(mcl_idx)

    print all_hits.head()
    print all_hits.dtypes

    return all_hits,max_mcl_idx

############ MAIN FUNCTION ##################
def interpolation_check(all_hits_df,coef_dict_array,precision_adjust=1e-5):
    '''
    DESCRIPTION:
        This function will
    '''
    #To hold the required property of cluster for out Check
    cluster_properties={}
    # properties={'init_energy':0,
    #             'init_Wx':0,
    #             'init_Wy':0,
    #             'init_Wz':0}

    #Iteratting layer by layer
    for layer,layer_hits in all_hits_df.groupby(['layer']):
        print '>>> Interpolating for Layer: %s',%(layer)

        #Getting the center of the cells which have hits
        layer_z_value=layer_hits[0,'z'] #will be same for all
        center_arr=layer_hits[['x','y']].values
        energy_arr=layer_hits[['energy']].values
        cluster3d_arr=layer_hits[['cluster3d']].values

        #Making the center as tuple for searching keys in coef_dict
        print '>>> Tuplizing the center of hits '
        hit_centers=[(center_arr[i,0],center_arr[i,1])
                        for i in range(center_arr.shape[0])]
        hex_centers=coef_dict_array[layer-1].keys()
        print '>>> Building the KDTree of Hex-Cells Center'
        hex_tree=cKDTree(hex_centers,balanced_tree=True)

        #Searching for hex cell corresp. of the hit in the hex cell tree
        print '>>> Querying the Tree for corresponding cells'
        indices=hex_tree.query_ball_point(hit_centers,r=precision_adjust)

        #Finally Calculating the Interpolation check values
        print '>>> Calculating the Multi-Cluster Properties'
        for hit_id in range(energy_arr.shape[0]):
            hex_cell_index=indices[hit_id]
            if not len(hex_cell_index)==1:
                print 'Multiple/No Hex Cell Matching with hit cell'
                sys.exit(1)
            hex_cell_center=hex_centers[hex_cell_index[0]]
            overlaps=coef_dict_array[layer][hex_cell_center]
            norm_coef=np.sum([overlap[1] for overlap in overlaps])

            #now filling the initial hex properties
            cluster3d=clusterd_arr[hit_id]
            if cluster3d not in cluster_properties.keys():
                init_list=np.array([0,0,0,0],dtype=np.float64)
                mesh_list=np.array([0,0,0,0],dtype=np.float64)
                cluster_properties[cluster3d]=(init_list,mesh_list)

            init_list=[energy_arr[hit_id],hex_cell_center[0],
                        hex_cell_center[1],layer_z_value]
            mesh_list=[energy]






if __name__=='__main__':
    #Reading the datafile and coef_file
    all_hits,max_mcl_idx=readDataFile(dfname)
    coef_dict_array=readCoefFile(cfname)

    #Now checking all the multicluster
    for i in range(max_mcl_idx+1):
        subset=all_hits[all_hits['cluster3d']==i][['layer','energy','x','y','z']]
        data_array=subset.values
        interpolation_check(i,data_array)
