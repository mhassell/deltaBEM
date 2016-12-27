import CalderonCalculusMatrices as CCM
import geometry
import matplotlib.pyplot as plt
import time

"""
Test module for the deltaBEM matrices.  Needs much improvement.
Last modified: December 27, 2016
"""

N = 100

g  = geometry.kite(N,0)
MATS = CCM.CalderonCalculusMatrices(g)

print "Without the fork: "

print "Quadrature matrix"
plt.spy(MATS[0])
plt.show()
time.sleep(1)
plt.close()

print "Mass matrix"
plt.spy(MATS[1])
plt.show()
time.sleep(1)
plt.close()

print "Pp"
print MATS[2]
time.sleep(1)

print "Pm"
print MATS[3]
time.sleep(1)

print "With the fork"

MATS = CCM.CalderonCalculusMatrices(g,1)

print "Quadrature matrix"
plt.spy(MATS[0])
plt.show()
time.sleep(1)
plt.close()

print "Mass matrix"
plt.spy(MATS[1])
plt.show()
time.sleep(1)
plt.close()

print "Pp"
plt.spy(MATS[2])
plt.show()
time.sleep(1)
plt.close()

print "Pm"
plt.spy(MATS[3])
plt.show()
time.sleep(1)
plt.close()









