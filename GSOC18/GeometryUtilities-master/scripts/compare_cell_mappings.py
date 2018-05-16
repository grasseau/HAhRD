#! /usr/bin/env python

import cPickle as pickle
from geometry.mapper import check_mappings_consistency



def main(file1, file2):
    mapping1 = pickle.load(open(file1, 'rb'))
    mapping2 = pickle.load(open(file2, 'rb'))
    check_mappings_consistency(mapping1, mapping2)



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--file1', dest='file1', help='First file to compare')
    parser.add_option('--file2', dest='file2', help='Second file to compare')
    (opt, args) = parser.parse_args()
    if not opt.file1 or not opt.file2:
        parser.print_help()
        print 'Error: Missing input file name'
        sys.exit(1)
    main(opt.file1, opt.file2)
