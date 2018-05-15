import math
from copy import deepcopy
import attr
from attr.validators import instance_of
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.affinity import rotate as shapely_rotate 


tan30 = math.tan(math.radians(30))

def hexagon(edge):
    vertices = []
    diameter = edge/tan30
    vx = -diameter/2.
    vy = -edge/2.
    for angle in range(0, 360, 60):
        vy += math.cos(math.radians(angle)) * edge
        vx += math.sin(math.radians(angle)) * edge
        vertices.append((vx,vy))
    return vertices

@attr.s
class Cell(object):
    # id variables
    id = attr.ib(validator=instance_of(int))
    layer = attr.ib(validator=instance_of(int))
    subdet = attr.ib(validator=instance_of(int))
    zside = attr.ib(validator=instance_of(int))
    module = attr.ib(validator=instance_of(int))
    # position and shape variables
    center = attr.ib(validator=instance_of(Point))
    vertices = attr.ib(validator=instance_of(Polygon))
    ieta = attr.ib(validator=instance_of(int), default=0)
    iphi = attr.ib(validator=instance_of(int), default=0)
    cell = attr.ib(validator=instance_of(int), default=0)

def rotate(cell, angle, origin='center'):
    rotated_cell = deepcopy(cell)
    rotated_cell.center = shapely_rotate(cell.center, angle, origin)
    rotated_cell.vertices = shapely_rotate(cell.vertices, angle, origin)
    return rotated_cell

def merge(cells):
    margin = min([cell.vertices.length for cell in cells])/1000.
    # inflate cells to make sure neighbour cells are covering each other
    cells_inflated = []
    for cell in cells:
        cells_inflated.append(cell.vertices.buffer(margin))
    # Merge cells 
    mergedcell = cells_inflated[0] 
    for cell in cells_inflated[1:]:
        mergedcell = mergedcell.union(cell)
    # erode the trigger cell to go back to the original cell sizes
    mergedcell = mergedcell.buffer(-margin)
    if mergedcell.__class__==MultiPolygon:
        mergedcell = mergedcell[0]
    return mergedcell
