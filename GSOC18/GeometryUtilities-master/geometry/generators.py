import math
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate, rotate

class PanelGenerator(object):
    def __init__(self, hexagon_size, length=3, width=2):
        v0 = Point((0,0))
        v1 = Point((0,hexagon_size*(length-1)))
        v2 = Point((0,hexagon_size*(length+width-2)))
        v2 = rotate(v2, 120, origin=v1)
        v3 = Point((0,hexagon_size*(width-1)))
        v3 = rotate(v3, 120, origin=v0)
        self._vertices = [(v0.x,v0.y),(v1.x,v1.y),(v2.x,v2.y),(v3.x,v3.y)]
        # create mirrored panel
        v2_mirror = rotate(v2, -240, origin=v1)
        v3_mirror = rotate(v3, -240, origin=v0)
        self._vertices_mirror = [(v2_mirror.x,v2_mirror.y),(v3_mirror.x,v3_mirror.y),(v0.x,v0.y),(v1.x,v1.y)]

    def __call__(self, point, mirror=False, rotation=0):
        panel = Polygon([(point.x+vx, point.y+vy) for vx,vy in self._vertices])
        if mirror:
            panel = Polygon([(point.x+vx, point.y+vy) for vx,vy in self._vertices_mirror])
        if rotation!=0:
            # FIXME: better to  rotate around vertex 0 than the center
            # of the polygon
            panel = rotate(panel, rotation)
        return panel


class PanelGeneratorTest(object):
    def __init__(self, hexagon_size, module_list=[]):
        self._vertices = []
        for module in module_list:
            column = module[0]
            row = module[1]
            point0 = Point((0,0))
            point0 = translate(point0, xoff=-hexagon_size*math.sqrt(3)/2.*column, yoff=hexagon_size/2.*column)
            self._vertices.append((point0.x, point0.y+hexagon_size*row))
        self._vertices.append((self._vertices[0][0]-hexagon_size/10, self._vertices[0][1]))

    def __call__(self, point, mirror=False, rotation=0):
        panel = Polygon([(point.x+vx, point.y+vy) for vx,vy in self._vertices])
        return panel

class SectorGenerator(object):
    def __init__(self, hexagon_size, panel_length=3, panel_width=2,
            # Number of rows for each panel column
            panel_rows = [4,4,3,3,2,1],
            # Type of panels in each of the columns
            panel_mirrored = [False, True, True, True, True, True]):
        panel_generator = PanelGenerator(hexagon_size, panel_length, panel_width)
        self._panels = []
        # Starting the first column at 0,0
        point0 = Point((0,0))
        point = point0
        # Loop over panel columns
        # Each row has a given length and is made of
        # mirrored or non-mirrored panels
        for rows,mirrored in zip(panel_rows, panel_mirrored):
            for row in xrange(rows):
                panel = panel_generator(point, mirror=mirrored)
                self._panels.append(panel)
                point = translate(point,xoff=0, yoff=hexagon_size*panel_length)
            # Translate the starting point for the next column
            if mirrored:
                point0 = translate(point0, xoff=-hexagon_size*math.sqrt(3), yoff=hexagon_size)
            else:
                point0 = translate(point0, xoff=-hexagon_size*math.sqrt(3)/2.*3., yoff=hexagon_size/2.)
            point = point0

    def __call__(self, point, rotation=0):
        panels = []
        for panel in self._panels:
            p = translate(panel, xoff=point.x, yoff=point.y)
            if rotation!=0:
                # Rotate around the origin
                p = rotate(p, rotation, origin=(0,0))
            panels.append(p)
        return panels


class SectorGeneratorTest(object):
    def __init__(self, hexagon_size, panel_list=[]):
        self._panels = []
        point0 = Point((0,0))
        for panel in panel_list:
            panel_generator = PanelGeneratorTest(hexagon_size, panel)
            panel = panel_generator(point0)
            self._panels.append(panel)

    def __call__(self, point, rotation=0):
        panels = []
        for panel in self._panels:
            p = translate(panel, xoff=point.x, yoff=point.y)
            if rotation!=0:
                # Rotate around the origin
                p = rotate(p, rotation, origin=(0,0))
            panels.append(p)
        return panels


class HexagonGenerator(object):
    def __init__(self, edge_length):
        self._vertices = []
        diameter = edge_length/math.tan(math.radians(30))
        vx = -diameter/2.
        vy = -edge_length/2.
        for angle in range(0, 360, 60):
            vy += math.cos(math.radians(angle)) * edge_length
            vx += math.sin(math.radians(angle)) * edge_length
            self._vertices.append((vx,vy))

    @property
    def vertices(self):
        return self._vertices

    def __call__(self, point, rotation=0):
        hexagon = Polygon([(point.x+vx, point.y+vy) for vx,vy in self.vertices])
        if rotation!=0:
            hexagon = rotate(hexagon, rotation)
        return hexagon


sqrt3o2 = math.sqrt(3)/2.
class GridGenerator(object):
    def __init__(self, shape, size):
        self.shape = shape
        self.size = size

    def diamond(self, point, step):
        return [
            Point((point.x + i*step+j*step/2., 
                point.y + j*step*sqrt3o2))
            for j in xrange(self.size) 
            for i in xrange(self.size)
            ]

    def square(self, point, step):
        return [
            Point((point.x + i*step*sqrt3o2, 
                point.y + j*step - (i%2)*step/2.))
            for i in xrange(self.size) 
            for j in xrange(self.size+(i%2))
            ]

    def hexagon(self, point, step):
        # Center of the grid
        points = [point]
        for i in xrange(self.size):
            # First generate the 6 corners of the hexagonal ring
            hex_generator = HexagonGenerator(step*(i+1))
            hex_points = hex_generator(point).exterior.coords[:]
            hex_reordered_points = []
            if i>0:
                # If the ring is incomplete, fill the missing points
                # Take pairs of closeby corners and fill the space in between
                for pair in zip(hex_points[:-1], hex_points[1:]):
                    hex_reordered_points.append(pair[0])
                    line = LineString(list(pair))
                    for split in xrange(1,i+1):
                        p = line.interpolate(line.length*float(split)/float(i+1))
                        #  hex_reordered_points.insert(-2, p.coords[0])
                        hex_reordered_points.append(p.coords[0])
                hex_reordered_points.append(hex_points[-1])
                hex_points = hex_reordered_points
            points.extend([Point(p) for p in hex_points[:-1]])
        return points



    def __call__(self, point, step, rotation=0):
        grid = []
        if self.shape=='diamond':
            grid = self.diamond(point, step)
        elif self.shape=='square':
            grid = self.square(point, step)
        elif self.shape=='hexagon':
            grid = self.hexagon(point, step)
        if rotation!=0:
            grid = [rotate(point,rotation,origin=point) for point in grid]
        return grid
            
        

def delete_point(polygon, i):
    coords = polygon.exterior.coords[:-1]
    del coords[i]
    return Polygon(coords)


def shift_point(polygon, i, shift):

    coords = polygon.exterior.coords[:-1]
    point = coords[i]
    coords[i] = (point[0]+shift[0], point[1]+shift[1])
    return Polygon(coords)

