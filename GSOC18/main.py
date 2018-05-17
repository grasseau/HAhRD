from hexCells_to_squareCell_interpolation import linear_interpolate_hex_to_square
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Polygon
from descartes.patch import PolygonPatch
import datetime

def plot_sq_cells(cell_d):
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
    t0=datetime.datetime.now()
    # fig=plt.figure()
    # ax1=fig.add_subplot(111)

    for hex_id,sq_overlaps in coef.items():
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        hex_cell=hex_cells_dict[hex_id]
        poly=hex_cell.vertices
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
        t0=t1
        #ax1.set_xlim(-160, 160)
        #ax1.set_ylim(-160, 160)
        #ax1.set_aspect(1)
        plt.show()



if __name__=='__main__':

    base_path='/home/abhinav/Desktop/HAhRD/GSOC18/'
    ## Generating the overlapping coefficient
    hex_filename=base_path+'hex_cells_data/hex_cells_dict.pkl'
    fhandle=open(hex_filename,'rb')
    hex_cells_dict=pickle.load(fhandle)
    fhandle.close()
    resolution=(1000,1000)
    layer=1
    sq_coef=linear_interpolate_hex_to_square(hex_filename,
                                                layer,resolution)
    coef_filename=base_path+'sq_cells_data/coef_dict_layer_%s_res_%s.pkl'%(layer,
                                                            resolution[0])
    fhandle=open(coef_filename,'wb')
    pickle.dump(sq_coef,fhandle,protocol=pickle.HIGHEST_PROTOCOL)
    fhandle.close()


    ## Plotting the sq cell for verification
    sq_filename=base_path+'sq_cells_data/sq_cells_dict_layer_%s_res_%s.pkl'%(layer,
                                                            resolution[0])
    fhandle=open(sq_filename,'rb')
    sq_cells_dict=pickle.load(fhandle)
    fhandle.close()

    #plot_sq_cells(sq_cells_dict)
    plot_hex_to_square_map(sq_coef,hex_cells_dict,sq_cells_dict)
