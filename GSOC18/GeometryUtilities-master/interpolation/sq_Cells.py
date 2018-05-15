#Importing appropriate shapes to create square cells
from shapely.geometry import Point,box

#Class Definition
class sq_Cells():
    '''
    This class will serve as an equivalent to the Cells Class of
    (Hexagonal Cells) of the detector, though all the attributes
    of that class wont be present here.
    '''
    def __init__(self,id,center,x_length,y_length):
        #Adding a unique id to the cells. Starting from top most
        self.id=id

        #Adding a Point object as the center of the square cells
        self.center=Point(center)

        #Now creating a Polygon(box) object as the square cells
        minx=center[0]-x_length/2
        maxx=center[0]+x_length/2
        miny=center[1]-y_length/2
        maxy=center[1]+y_length/2
        self.polygon=box(minx,miny,maxx,maxy)
        #should we keep the name polygon or vertices as in
        #JB's Script for consistency
