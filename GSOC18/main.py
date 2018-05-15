from hexCells_to_squareCell_interpolation import linear_interpolate_hex_to_square
import sys
sys.path.append('/home/abhinav/Desktop/HAhRD/GSOC18/cell_data/cells_out.pkl')

if __name__=='__main__':
    filename='/home/abhinav/Desktop/HAhRD/GSOC18/cell_data/cells_out.pkl'
    resolution=(100,100)
    layer=1
    linear_interpolate_hex_to_square(filename,layer,resolution)
