import math
import numpy as np
from root_numpy import root2array
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from geometry.cell import Cell, hexagon
from geometry.generators import HexagonGenerator, GridGenerator, delete_point, shift_point

# Constants
cos60 = math.cos(math.radians(60))
sin60 = math.sin(math.radians(60))
sqrt3t2o9 = 2.*math.sqrt(3)/9.

# cell parameters
large_cells  = {
    'cell_corner_size':0.6492566208977884,
    'half_cells_edge_topleft':[107,118,126,131],
    'half_cells_edge_topright':[132,130,125,117],
    'half_cells_edge_right':[106,83,60,37],
    'half_cells_edge_bottomright':[25,14,6,1],
    'half_cells_edge_bottomleft':[0,2,7,15],
    'half_cells_edge_left':[26,49,72,95],
}

small_cells  = {
    'cell_corner_size':0.4761215219917115,
    'half_cells_edge_topleft':[200,214,225,233,238],
    'half_cells_edge_topright':[239,237,232,224,213],
    'half_cells_edge_right':[184,153,122,91,60],
    'half_cells_edge_bottomright':[44,29,17,8,2],
    'half_cells_edge_bottomleft':[0,3,9,18,30],
    'half_cells_edge_left':[45,76,107,138,169],
}


cell_offset = 0
cell_mask = 0xFF
wafer_offset = 8
wafer_mask = 0x3FF

# cell ID manipulation functions
def compute_id(wafer, cell):
    id = 0
    id |= ((cell & cell_mask) << cell_offset);
    id |= ((wafer & wafer_mask) << wafer_offset);
    return id

def cell_id(id):
    return id & cell_mask

def wafer_id(id):
    return (id>>wafer_offset) & wafer_mask




class CenterCorrector(object):
    def __init__(self, wafertype):
        self.wafertype = wafertype
        self.params = small_cells if wafertype==1 else large_cells
        self.edge = self.params['cell_corner_size']

    def shift_leftborderhalfcell(self, point):
        return translate(point, xoff=-self.edge*sqrt3t2o9, yoff=0.)

    def shift_topleftborderhalfcell(self, point):
        return translate(point, xoff=-self.edge*sqrt3t2o9*cos60, yoff=self.edge*sqrt3t2o9*sin60)

    def shift_toprightborderhalfcell(self, point):
        return translate(point, xoff=self.edge*sqrt3t2o9*cos60, yoff=self.edge*sqrt3t2o9*sin60)

    def shift_bottomrightborderhalfcell(self, point):
        return translate(point, xoff=self.edge*sqrt3t2o9*cos60, yoff=-self.edge*sqrt3t2o9*sin60)

    def shift_bottomleftborderhalfcell(self, point):
        return translate(point, xoff=-self.edge*sqrt3t2o9*cos60, yoff=-self.edge*sqrt3t2o9*sin60)

    def shift_rightborderhalfcell(self, point):
        return translate(point, xoff=self.edge*sqrt3t2o9, yoff=0.)

    def __call__(self, index):
        if index in self.params['half_cells_edge_left']:
            return self.shift_leftborderhalfcell
        elif index in self.params['half_cells_edge_topleft']:
            return self.shift_topleftborderhalfcell
        elif index in self.params['half_cells_edge_topright']:
            return self.shift_toprightborderhalfcell
        elif index in self.params['half_cells_edge_bottomright']:
            return self.shift_bottomrightborderhalfcell
        elif index in self.params['half_cells_edge_bottomleft']:
            return self.shift_bottomleftborderhalfcell
        elif index in self.params['half_cells_edge_right']:
            return self.shift_rightborderhalfcell
        return lambda point:point


class CellTransform(object):
    def __init__(self, wafertype):
        self.wafertype = wafertype
        self.params = small_cells if wafertype==1 else large_cells

    def transform_leftborderhalfcell(self, polygon):
        polygon = delete_point(polygon, 0)
        polygon = delete_point(polygon, -1)
        return polygon

    def transform_topleftborderhalfcell(self, polygon):
        polygon = delete_point(polygon, 0)
        polygon = delete_point(polygon, 0)
        return polygon

    def transform_toprightborderhalfcell(self, polygon):
        polygon = delete_point(polygon, 1)
        polygon = delete_point(polygon, 1)
        return polygon

    def transform_bottomrightborderhalfcell(self, polygon):
        polygon = delete_point(polygon, 3)
        polygon = delete_point(polygon, 3)
        return polygon

    def transform_bottomleftborderhalfcell(self, polygon):
        polygon = delete_point(polygon, 4)
        polygon = delete_point(polygon, 4)
        return polygon

    def transform_rightborderhalfcell(self, polygon):
        polygon = delete_point(polygon, 2)
        polygon = delete_point(polygon, 2)
        return polygon

    def __call__(self, index):
        if index in self.params['half_cells_edge_left']:
            return self.transform_leftborderhalfcell
        elif index in self.params['half_cells_edge_topleft']:
            return self.transform_topleftborderhalfcell
        elif index in self.params['half_cells_edge_topright']:
            return self.transform_toprightborderhalfcell
        elif index in self.params['half_cells_edge_bottomright']:
            return self.transform_bottomrightborderhalfcell
        elif index in self.params['half_cells_edge_bottomleft']:
            return self.transform_bottomleftborderhalfcell
        elif index in self.params['half_cells_edge_right']:
            return self.transform_rightborderhalfcell
        return lambda polygon:polygon

# pre-compute hexagon vertices
hexagon_generator = {
        -1:HexagonGenerator(large_cells['cell_corner_size']),
        1:HexagonGenerator(small_cells['cell_corner_size']),
        }

def cell_vertices(x, y, wafertype, cell):
    # Shift cell barycenter to hexagon center for half cells
    point_shifter = CenterCorrector(wafertype)
    cell_center = point_shifter(cell)(Point((x,y)))
    # Create vertices of the cell
    cell_transform = CellTransform(wafertype)
    vertices = cell_transform(cell)(hexagon_generator[wafertype](cell_center))
    return vertices

def read_geometry(filename, treename, subdet, layer, wafer=-1):
    # Read cells from one layer
    selection = "zside==1 && layer=={0} && subdet=={1}".format(layer,subdet)
    if wafer!=-1:
        selection += ' && wafer=={}'.format(wafer)
    branches = ['id',
            'wafer', 'wafertype', 'cell', 
            'x', 'y']
    cells = root2array(filename, treename=treename, branches=branches, selection=selection)
    # Create cell shapes
    output_cells = []
    for cell in cells:
        vertices = cell_vertices(cell['x'], cell['y'], cell['wafertype'], cell['cell']) 
        barycenter = Point((cell['x'],cell['y']))
        output_cells.append(Cell(
            #  id=int(cell['id']),
            id=int(compute_id(cell['wafer'], cell['cell'])),
            layer=layer,
            subdet=subdet,
            zside=1,
            module=int(cell['wafer']),
            center=barycenter,
            vertices=vertices
            ))
    return output_cells

def read_bh_geometry(filename, treename):
    # Read cells from one side
    selection = "zside==1 && subdet==2 && layer==1"
    branches = ['id',
            'ieta', 'iphi', 
            'x', 'y']
    for corner in xrange(1,5):
        branches.append('x{}'.format(corner))
        branches.append('y{}'.format(corner))
    cells = root2array(filename, treename=treename, branches=branches, selection=selection)
    # Create cell shapes
    output_cells = []
    for cell in cells:
        vertices = Polygon([(cell['x1'],cell['y1']),
            (cell['x2'],cell['y2']),
            (cell['x3'],cell['y3']),
            (cell['x4'],cell['y4'])])
        barycenter = Point((cell['x'],cell['y']))
        output_cells.append(Cell(
            id=int(cell['id']),
            layer=1,
            subdet=5,
            zside=1,
            module=1,
            ieta=int(cell['ieta']),
            iphi=int(cell['iphi']),
            center=barycenter,
            vertices=vertices
            ))
    return output_cells



