deltaBEM in Python

I'd like to pull some deltaBEM tools into Python, since deltaBEM is quite straighforward to implement and works quite well.

My ultimate goal is to learn more about Numpy, since I am already quite comfortable in Matlab.  I'd like to learn about the similarities and differences.

This isn't meant to supercede or in any way replace the original Matlab implementation.  
This is just a learning tool for me that has some pretty cool capabilities.
If you're reading this and would like to use deltaBEM for research, check out the
Matlab implementation at www.math.udel.edu/~fjsayas/deltaBEM.


TBD:

1. Implement core functions: operators, and potentials as function handles (lambdas) of the complex paramter s.

2. Once steady state and time-harmonic methods are implemented, develop CQ code to move to time domain.

3. Include convergence studies to show the correctness of the methods and plots/movies to learn how to use matplotlib and the like.

4. Need to figure out a replacement for the triangulateGeometry method (this uses PDEtool in Matlab).

5. In CalderonCalculusMatrices.py and laplace.py need to fix using Pp and Pm with np.dot.  This will not work if Pp and Pm are sparse, and is pointless if they are scalar.

6. Need affine translations of geometries (not including inversions)
