#! /usr/bin/env python

import datetime
import numpy as np

ietamin = 17
ietamax = 100
iphimin = 1
iphimax = 360

triggercellsize = 3
modulephisize = 10
moduleetasize = (ietamax-ietamin+1)/3/2

nmodulephi = (iphimax-iphimin+1)/triggercellsize/modulephisize


def main(output_file):
    mapping  = {}
    for iphi in xrange(iphimin, iphimax+1):
        iphi_tc = (iphi-iphimin)/triggercellsize+1
        iphi_mod = (iphi_tc-1)/modulephisize+1
        iphi_mod_cell = (iphi_tc-1)%modulephisize+1
        for ieta in xrange(ietamin, ietamax+1):
            ieta_tc = (ieta-ietamin)/triggercellsize+1
            ieta_mod = (ieta_tc-1)/moduleetasize+1
            ieta_mod_cell = (ieta_tc-1)%moduleetasize+1
            module_id = iphi_mod + (ieta_mod-1)*nmodulephi
            tc_id = iphi_mod_cell + (ieta_mod_cell-1)*modulephisize
            mapping[(ieta,iphi)] = (module_id,tc_id)

    # Disconnect ieta=16
    for iphi in xrange(1,73):
        mapping[(16,iphi)] = (0,iphi)

    # Save mapping
    with open(output_file, 'w') as f:
        for cell, tc in sorted(mapping.items()):
            print >>f, cell[0], cell[1], tc[0], tc[1]



if __name__=='__main__':
    import sys
    import optparse
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--output', dest='output_file', help='Output pickle file', default='mapping.pkl')
    (opt, args) = parser.parse_args()
    main(opt.output_file)
