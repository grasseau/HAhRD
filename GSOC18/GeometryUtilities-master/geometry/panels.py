
import cPickle as pickle
import math
from geometry.cmssw import read_geometry
from geometry.generators import HexagonGenerator, GridGenerator, shift_point, SectorGenerator, SectorGeneratorTest
from shapely.geometry import Polygon, Point
from geometry.cell import Cell
from shapely.affinity import rotate


# Constants
sqrt3o2 = math.sqrt(3)/2.

panel_offset = 0
panel_mask = 0x1F
sector_offset = 5
sector_mask = 0x7

# panel ID manipulation functions
def compute_id(sector, panel):
    id = 0
    id |= ((panel & panel_mask) << panel_offset);
    id |= ((sector & sector_mask) << sector_offset);
    return id

def panel_id(id):
    return id & panel_mask

def sector_id(id):
    return (id>>sector_offset) & sector_mask



def intersect_modules(polygon, modules):
    module_intersect = []
    for module in modules:
        if polygon.intersection(module.vertices).area>0.:
            module_intersect.append(module)
    return module_intersect


def generate_modules(wafer_size, grid_size):
    grid_generator = GridGenerator('hexagon', grid_size)
    module_centers = grid_generator(point=Point((0,0)), step=wafer_size*sqrt3o2)
    hex_generator = HexagonGenerator(wafer_size/2.)
    module_vertices = [rotate(hex_generator(point), 30, point) for i,point in enumerate(module_centers)]
    modules = []
    for i,(vertices,center) in enumerate(zip(module_vertices,module_centers)):
        modules.append(Cell(
            id=i,
            layer=1,
            zside=1,
            subdet=3,
            module=i,
            center=center,
            vertices=vertices
            ))
    return modules

def generate_sectors(full_layer, wafer_size):
    sector = Polygon([(0,0)]+list(full_layer.exterior.coords)[:2])
    sector = shift_point(sector, 0, (0,wafer_size*sqrt3o2))
    sector = shift_point(sector, 1, (0,wafer_size*sqrt3o2))
    sectors = [sector]
    sectors.append(rotate(sector, 60, origin=(0,0)))
    sectors.append(rotate(sector, 120, origin=(0,0)))
    sectors.append(rotate(sector, 180, origin=(0,0)))
    sectors.append(rotate(sector, 240, origin=(0,0)))
    sectors.append(rotate(sector, 300, origin=(0,0)))
    return sectors

def generate_panels(wafer_size):
    panels0 = SectorGenerator(wafer_size*sqrt3o2)(Point(0,wafer_size*sqrt3o2*2))
    panels = []
    panels.append(panels0)
    panels.append([rotate(panel, 60, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 120, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 180, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 240, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 300, origin=(0,0)) for panel in panels0])
    return panels


def generate_panels_test(wafer_size, panel_list):
    panels0 = SectorGeneratorTest(wafer_size*sqrt3o2, panel_list)(Point(0,0))
    panels = []
    panels.append(panels0)
    panels.append([rotate(panel, 60, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 120, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 180, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 240, origin=(0,0)) for panel in panels0])
    panels.append([rotate(panel, 300, origin=(0,0)) for panel in panels0])
    return panels

def modules_to_panels(wafer_size, grid_size):
    modules = generate_modules(wafer_size, grid_size)
    full_layer = HexagonGenerator(wafer_size*grid_size*sqrt3o2)(Point(0,0))
    sectors = generate_sectors(full_layer, wafer_size)
    panels = generate_panels(wafer_size)
    sector_to_modules = {}
    module_to_panel = {}
    panel_to_modules = {}
    for i,sector in enumerate(sectors):
        sector_to_modules[i] = intersect_modules(sector, modules)
    for isec,sector_panels in enumerate(panels):
        sector_modules = sector_to_modules[isec]
        for ipan,panel in enumerate(sector_panels):
            panel_to_modules[compute_id(isec,ipan+1)] = []
            panel_modules = intersect_modules(panel, sector_modules)
            for module in panel_modules:
                module_to_panel[module.id] = (isec, ipan+1)
                panel_to_modules[compute_id(isec,ipan+1)].append(module.id)
    return module_to_panel, panel_to_modules


def modules_to_panels_test(wafer_size, grid_size, panel_list):
    modules = generate_modules(wafer_size, grid_size)
    full_layer = HexagonGenerator(wafer_size*grid_size*sqrt3o2)(Point(0,0))
    sectors = generate_sectors(full_layer, wafer_size)
    panels = generate_panels_test(wafer_size, panel_list)
    sector_to_modules = {}
    module_to_panel = {}
    panel_to_modules = {}
    for i,sector in enumerate(sectors):
        sector_to_modules[i] = intersect_modules(sector, modules)
    for isec,sector_panels in enumerate(panels):
        sector_modules = sector_to_modules[isec]
        for ipan,panel in enumerate(sector_panels):
            panel_to_modules[compute_id(isec,ipan+1)] = []
            panel_modules = intersect_modules(panel, sector_modules)
            for module in panel_modules:
                module_to_panel[module.id] = (isec, ipan+1)
                panel_to_modules[compute_id(isec,ipan+1)].append(module.id)
    return module_to_panel, panel_to_modules
