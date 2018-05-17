#! /usr/bin/env python

import datetime
import cPickle as pickle
from geometry.zoltan_split import module_grid
from geometry.cmssw import read_geometry
from geometry.mapper import map_cells
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Polygon
from descartes.patch import PolygonPatch

"""
Terminology:

1 - The sufixe _x specifify (if required) the type of the variable
    Example :
    cells_d is the dictionnary form of the cells objects

"""

input_default_file = '/data_CMS/cms/grasseau/HAhRD/test_triggergeom.root'

"""
Read the 'cells' of the HGCAL geometry (old release)
  Parameters :
    input_file ...
    layer ...
    subdet: 
      3: EE (ECAL, Silicon)
      4: FH (Front HCAL, Silicon)
      5: BH (Back HCAL, Scintillator) 
  return a dictionnary in which the keys are the cell IDs
"""
def readGeometry( input_file,  layer, subdet ):
    t0 = datetime.datetime.now()
    treename = 'hgcaltriggergeomtester/TreeCells'
    cells = read_geometry(filename=input_file, treename=treename, 
              subdet=subdet, layer=layer, wafer=-1)
    cells_d = dict([(c.id, c) for c in cells])
    t1 = datetime.datetime.now()
    print 'Cells read: number=', len(cells), ', time=', t1-t0
    return cells_d
"""
Arguments:
 cells_d: ...
 rate: cell fractions plotted [0, 1]
"""
def simple_plane_plot( cells_d, fraction ):
  t0 = datetime.datetime.now()
  fig=plt.figure()
  ax1=fig.add_subplot(111)
  i=0
  period = int(  1.0 / fraction )
  for id,cell in cells_d.items():
    if ( i % period == 0):
      poly=cell.vertices
      # Plot vertices
      # commented, save CPU time
      # x,y=poly.exterior.xy
      # ax1.plot(x,y,'o',zorder=1)
      patch=PolygonPatch(poly,alpha=0.5,zorder=2, edgecolor='blue')
      ax1.add_patch(patch)
    i+=1
    #plt.show()
    #if i%100==0:
    #    plt.savefig(str(i)+'.png')
    #plt.show()   

  t1 = datetime.datetime.now()
  print 'Cells plotted: number=', int(fraction * len(cells_d)), ', time=', t1-t0
  ax1.set_xlim(-160, 160)
  ax1.set_ylim(-160, 160)
  ax1.set_aspect(1)
  plt.show()

if __name__=='__main__':
  import sys
  import optparse
  usage = 'usage: %prog [options]'
  parser = optparse.OptionParser(usage)
  parser.add_option('--input_geometry', dest='input_file', help='Input geometry file', default=input_default_file)
  # Not used
  # parser.add_option('--output', dest='output_file', help='Output pickle file', default='mapping.pkl')
  parser.add_option('--layer', dest='layer', help='Layer to be mapped', type='int', default=1)
  parser.add_option('--subdet', dest='subdet', help='Subdet', type='int', default=3)
  (opt, args) = parser.parse_args()
  if not opt.input_file:
    parser.print_help()
    print 'Error: Missing input geometry file name'
    sys.exit(1)
  cells_d = readGeometry( opt.input_file, opt.layer, opt.subdet )
  simple_plane_plot( cells_d, 0.20 ) 
