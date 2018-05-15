#! /usr/bin/env python

import datetime
import cPickle as pickle
import numpy as np
from geometry.panels import generate_modules, modules_to_panels


def main(output_file):
    # Produce Zoltan/Split trigger cells
    # 8" flat to flat distance is 164.9mm -> 190.41mm vertex to vertex 
    print 'Mapping modules and panels'
    t0 = datetime.datetime.now()
    module_to_panel, panel_to_modules = modules_to_panels(wafer_size=19.041, grid_size=13)
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
