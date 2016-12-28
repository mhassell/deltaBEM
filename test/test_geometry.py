"""
A script to test the geometry module, many improvements needed

Last Modified: December 27, 2016
"""

import geometry

N = 100

g = geometry.kite(100,0)

print "Number of points: {}".format(N)
print "Midpoint shape: {}".format(g['midpt'].shape)
print "Breakpoint shape: {}".format(g['brkpt'].shape)
print "derivative shape: {}".format(g['xp'].shape)
print "normal deriv shape: {}".format(g['normal'].shape)
print "next vector shape: {}".format(g['next'].shape)
print "component: {}".format(g['comp'])

print g['midpt'][:,0].shape
print g['midpt'][0,:].shape
