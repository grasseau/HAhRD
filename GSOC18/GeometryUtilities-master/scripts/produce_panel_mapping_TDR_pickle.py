#! /usr/bin/env python

import datetime
import cPickle as pickle
import numpy as np
from geometry.panels import generate_modules, modules_to_panels_test

panel_list = []
# 1st column
panel_list.append([(0,2),(0,3),(0,4)])
panel_list.append([(0,5),(0,6),(0,7)])
panel_list.append([(0,8),(0,9),(0,10),(0,11)])
# 2nd column
panel_list.append([(1,2),(1,3),(2,2)])
panel_list.append([(1,4),(1,5),(1,6)])
panel_list.append([(1,7),(1,8),(1,9),(1,10)])
# 3rd column
panel_list.append([(2,3),(2,4),(2,5)])
panel_list.append([(2,6),(2,7),(2,8),(2,9),(2,10)])
# +4 column
panel_list.append([(3,3),(3,4),(3,5)])
panel_list.append([(3,6),(3,7),(3,8),(3,9)])
# -1 column
panel_list.append([(1,1),(2,1),(3,1)])
panel_list.append([(4,1),(5,1),(6,1)])
panel_list.append([(7,1),(8,1),(9,1),(10,1)])
# -2 column
panel_list.append([(3,2),(4,2),(5,2)])
panel_list.append([(6,2),(7,2),(8,2),(9,2),(10,2)])
# -3 column
panel_list.append([(4,3),(5,3),(6,3),(7,3),(8,3),(9,3)])
#panel_list.append([(7,3),(8,3),(9,3)])
# -4 column
panel_list.append([(4,4),(5,4),(6,4),(7,4),(8,4)])
# +5 column
panel_list.append([(4,5),(4,6),(4,7),(4,8)])
# end
panel_list.append([(5,5),(5,6),(5,7),(6,6),(7,5),(6,5)])
panel_list.append([(5,8),(6,7),(7,6),(8,5)])


def main(output_file):
    # Produce Zoltan/Split trigger cells
    # 8" flat to flat distance is 164.9mm -> 190.41mm vertex to vertex 
    print 'Mapping modules and panels'
    t0 = datetime.datetime.now()
    module_to_panel, panel_to_modules = modules_to_panels_test(wafer_size=19.041,
                                                                     grid_size=13,
                                                                     panel_list=panel_list
                                                                    )
    t1 = datetime.datetime.now()
    print '->', (t1-t0).seconds, 'seconds'
    # Save mapping
    pickle.dump(module_to_panel, open(output_file, 'wb'))



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--output', dest='output_file', help='Output pickle file', default='panel_mapping.pkl')
    (opt, args) = parser.parse_args()
    main(opt.output_file)
