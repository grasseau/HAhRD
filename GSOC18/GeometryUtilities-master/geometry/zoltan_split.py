import math
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from geometry.cell import Cell, hexagon, rotate
from geometry.generators import HexagonGenerator, GridGenerator, delete_point, shift_point


sqrt3o2 = math.sqrt(3)/2.
sqrt3t2 = math.sqrt(3)*2.
tan30 = math.tan(math.radians(30))
cos60 = math.cos(math.radians(60))
sin60 = math.sin(math.radians(60))

cell_offset = 0
cell_mask = 0xFF
third_offset = 8
third_mask = 0x3
module_offset = 10
module_mask = 0x3FF


# cell ID manipulation functions
def compute_id(module, third, cell):
    id = 0
    id |= ((cell & cell_mask) << cell_offset);
    id |= ((third & third_mask) << third_offset);
    id |= ((module & module_mask) << module_offset);
    return id

def cell_id(id):
    return id & cell_mask

def third_id(id):
    return (id>>third_offset) & third_mask


def module_id(id):
    return (id>>module_offset) & module_mask


class CellTransform(object):
    def __init__(self, cell_size, grid_size):
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.ncells = grid_size*grid_size
        # define sets of cells to be transformed
        self.smallbordercells = range(grid_size*(grid_size-1)+1,self.ncells-1)
        self.largebordercells = [i for i in xrange(self.ncells) if i%grid_size==0]
        self.largebordercells = self.largebordercells[:-1]
        self.cornerleft = [grid_size*(grid_size-1)]
        self.cornerright = [self.ncells-1]

    def transform_largebordercell(self, polygon):
        polygon = shift_point(polygon, 1, (-self.cell_size/sqrt3t2*sin60,self.cell_size/sqrt3t2*cos60)) 
        polygon = delete_point(polygon, 0)
        polygon = shift_point(polygon, -1, (-self.cell_size/sqrt3t2*sin60,self.cell_size/sqrt3t2*cos60)) 
        return polygon

    def transform_smallbordercell(self, polygon):
        polygon = delete_point(polygon, 1)
        return polygon

    def transform_cornerleft(self, polygon):
        polygon = delete_point(polygon, 1)
        polygon = shift_point(polygon, -1, (-self.cell_size/sqrt3t2*sin60,self.cell_size/sqrt3t2*cos60)) 
        return polygon

    def transform_cornerright(self, polygon):
        polygon = delete_point(polygon, 1)
        polygon = shift_point(polygon, 2, (self.cell_size/sqrt3t2*sin60,self.cell_size/sqrt3t2*cos60)) 
        return polygon

    def __call__(self, index):
        if index in self.smallbordercells:
            return self.transform_smallbordercell
        elif index in self.largebordercells:
            return self.transform_largebordercell
        elif index in self.cornerleft:
            return self.transform_cornerleft
        elif index in self.cornerright:
            return self.transform_cornerright
        return lambda polygon:polygon

def trigger_cells(cells, size=2):
    triggercells = []
    nrows = int(math.sqrt(len(cells)))
    if float(nrows)!=math.sqrt(len(cells)):
        raise RuntimeError('The cell grid used to create trigger cells is not squared')
    if nrows%size!=0:
        raise RuntimeError('The cell grid size is not a multiple of the trigger cell size')
    index_grid = np.arange(len(cells)).reshape((nrows,nrows))
    for i in range(0,nrows,size):
        for j in range(0,nrows,size):
            # Extract cell indices to be included in the trigger cell
            index_window = index_grid[i:i+size,j:j+size].flatten()
            # dilate cell to ensure coverage of neighbor cells
            margin = min(cells[i+nrows*j].length, cells[i+1+nrows*j].length)/1000.
            cells_window = []
            for index in index_window:
                cells_window.append(cells[index].buffer(margin))
            # Merge cells into one trigger cell
            triggercell = cells_window[0] 
            for cell in cells_window[1:]:
                triggercell = triggercell.union(cell)
            # erode the trigger cell to go back to the original cell sizes
            triggercell = triggercell.buffer(-margin)
            triggercells.append(triggercell)
    return triggercells


def module_third(wafer_size, ncells, module_center=Point((0,0)), module_id=0, triggercell_size=1):
    # Compute the cell grid length for 1/3 of a module
    grid_size = int(math.sqrt(ncells/3))
    # This geometry is of the rotated type
    # The wafer size below is the vertex to vertex distance
    # The cell size is the edge to edge distance
    cell_size = wafer_size/grid_size/2.
    # Create grid of cells along the usual hexagon axes (60deg rotated axes)
    grid_generator = GridGenerator('diamond', grid_size)
    reference_position = translate(module_center,
            xoff=-cell_size*(grid_size-1), 
            yoff=cell_size*tan30)
    cell_centers = grid_generator(reference_position, cell_size)
    # Create cells corresponding to 1/3 of a module
    hex_generator = HexagonGenerator(cell_size*tan30)
    cell_transform = CellTransform(cell_size, grid_size)
    cell_vertices = [cell_transform(i)(hex_generator(point)) for i,point in enumerate(cell_centers)]
    # Merge cells in trigger cells if requested
    if triggercell_size>1:
        cell_vertices = trigger_cells(cell_vertices, size=triggercell_size)
        cell_centers = [c.centroid for c in cell_vertices]
    cells = []
    for i,(vertices,center) in enumerate(zip(cell_vertices,cell_centers)):
        cells.append(Cell(
            id=compute_id(module=module_id, third=0, cell=i),
            layer=1,
            zside=1,
            subdet=3,
            module=module_id,
            center=center,
            vertices=vertices
            ))
    return cells


def module(wafer_size, ncells, center=Point((0,0)), module_id=0, triggercell_size=1):
    # create the cells for the first third
    cells_third0 = module_third(wafer_size, ncells, center, module_id, triggercell_size=triggercell_size)
    # create the two other thirds, rotated wrt the first one
    cells_third1 = [rotate(cell, 120, center) for cell in cells_third0]
    cells_third2 = [rotate(cell, 240, center) for cell in cells_third0]
    for c in cells_third1:
        c.id = compute_id(module=module_id, third=1, cell=cell_id(c.id))
    for c in cells_third2:
        c.id = compute_id(module=module_id, third=2, cell=cell_id(c.id))
    return cells_third0 + cells_third1 + cells_third2


def module_grid(wafer_size, ncells, grid_size=3, triggercell_size=1, center=(0,0)):
    # Create hexagonal grid of points corresponding to module centers
    grid_generator = GridGenerator('hexagon', grid_size)
    grid_shift = Point(center)
    module_centers = grid_generator(point=grid_shift, step=wafer_size*sqrt3o2)
    # Create a module at each grid point
    modules = [module(wafer_size, ncells, center=Point(center), module_id=i, triggercell_size=triggercell_size) for i,center in enumerate(module_centers)]
    return modules


