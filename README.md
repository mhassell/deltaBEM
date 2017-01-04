deltaBEM in Python

This isn't meant to supercede or in any way replace the original Matlab implementation.  
This is just a learning tool for me that has some pretty cool capabilities.
If you're reading this and would like to use deltaBEM for research, check out the
Matlab implementation at www.math.udel.edu/~fjsayas/deltaBEM.


TBD:

1. Convergence studies with CQ (SL is done, DL TBD)

2. Plots/movies to learn how to use matplotlib and the like (DONE! Could use some tweaking, but it works!)

3. Need to figure out a replacement for the triangulateGeometry.m method.  I have some notes on this floating around.

4. In CalderonCalculusMatrices.py, laplace.py, and helmholtz.py need to fix using Pp and Pm with np.dot.  This will not work if Pp and Pm are sparse, and is pointless if they are scalar.

5. Need affine translations of geometries (not including inversions)

6. Need to parallelize & symmetrize CQ code

6. Long term goal: need to make np.solve and np.dot compatible with scipy sparse matrices.  
