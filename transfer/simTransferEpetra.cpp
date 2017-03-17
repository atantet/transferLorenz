#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include <ergoGrid.hpp>
#include <transferOperator.hpp>
#include <gsl_extension.hpp>
#include "../cfg/readConfig.hpp"

// Include selected communicator class required by Epetra objects
#ifdef EPETRA_MPI
// Your code is an existing MPI code, so it presumably includes mpi.h directly.
#include <mpi.h>
// Epetra's wrapper for MPI_Comm.  This header file only exists if
// Epetra was built with MPI enabled.
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif
#include <Epetra_Map.h>
// Include header for Epetra sparse matrix and multivector.
#include <Epetra_CrsMatrix.h>
#include <Epetra_RowMatrixTransposer.h>
#include "Epetra_Time.h"
// HDF5 support
#include <EpetraExt_HDF5.h>


/** \file transfer.cpp
 *  \brief Get transition matrices and distributions directly from time series.
 *   
 * Get transition matrices and distributions from a long time series
 * (e.g. simulation output).
 * Takes as first a configuration file to be parsed with libconfig C++ library.
 * First read the observable and get its mean and standard deviation
 * used to adapt the grid.
 * A rectangular grid is used here.
 * A grid membership vector is calculated for each time series 
 * assigning to each realization a grid box.
 * Then, the membership matrix is calculated for a given lag.
 * The forward transition matrices are calculated from the membership matrix.
 * Finally, the results are printed.
 */

/** \brief Conversion from spherical to Cartesian coordinates. */
void sphere2Cart(gsl_vector *x);
/** \brief Conversion from Cartesian to spherical coordinates. */
void cart2Sphere(gsl_vector *x);
/** \brief Get final box of a trajectory. */
int getTraj(model *mod, Grid *grid, const gsl_vector *IC, const double tau,
	    const double dt);
/** \brief Get box boundaries. */
void getBoxBoundaries(size_t box0, gsl_vector_uint *nx,
		      gsl_vector_uint *multiIdx, Grid *grid,
		      gsl_vector *minBox, gsl_vector *maxBox)


/** \brief Calculate transfer operators from time series.
 *
 *  After parsing the configuration file,
 *  the time series is read and an observable is designed
 *  selecting components with a given embedding lag.
 *  A membership vector is then built from the observable,
 *  attaching the box they belong to to every realization.
 *  The membership vector is then converted to a membership
 *  matrix for different lags and the transfer operators 
 *  built. The results are then written to file.
 */
int main(int argc, char * argv[])
{
  // Read configuration file
  if (argc < 2)
    {
      std::cout << "Enter path to configuration file:" << std::endl;
      std::cin >> configFileName;
    }
  else
    {
      strcpy(configFileName, argv[1]);
    }
  try
    {
      Config cfg;
      std::cout << "Sparsing config file " << configFileName << std::endl;
      cfg.readFile(configFileName);
      readGeneral(&cfg);
      readModel(&cfg);
      readSimulation(&cfg);
      readSprinkle(&cfg);
      readGrid(&cfg);
      readTransfer(&cfg);
      std::cout << "Sparsing success.\n" << std::endl;
    }
  catch(const SettingTypeException &ex) {
    std::cerr << "Setting " << ex.getPath() << " type exception." << std::endl;
    throw ex;
  }
  catch(const SettingNotFoundException &ex) {
    std::cerr << "Setting " << ex.getPath() << " not found." << std::endl;
    throw ex;
  }
  catch(const SettingNameException &ex) {
    std::cerr << "Setting " << ex.getPath() << " name exception." << std::endl;
    throw ex;
  }
  catch(const ParseException &ex) {
    std::cerr << "Parse error at " << ex.getFile() << ":" << ex.getLine()
              << " - " << ex.getError() << std::endl;
    throw ex;
  }
  catch(const FileIOException &ex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    throw ex;
  }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  try {
#ifdef EPETRA_MPI
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Create an Epetra communicator
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
    Epetra_SerialComm Comm;
#endif
    const int MyPID = Comm.MyPID ();
    const int NumProc = Comm.NumProc ();
    // To check how many processes are ranb
    std::cout << "MyPID = " << MyPID << " / " << (NumProc - 1) << std::endl;
    
    // Grid declarations
    const double tau = gsl_vector_get(tauRng, 0);
    const int nTrajPerBox = nTraj / N;
    size_t nIn, nTot, boxf, traj;

    // Transfer operator declarations
    char forwardTransitionFileName[256], postfix[256];
    char srcPostfix[256], dstGridPostfix[256], gridFileName[256];
    sprintf(srcPostfix, "_%s", caseName);
    sprintf(gridFileName, "%s/grid/grid%s%s.txt", resDir, srcPostfix,
	    gridPostfix);
    sprintf(dstGridPostfix, "%s%s_rho%04d_L%d_dt%d_nTraj%d",
	    srcPostfix, gridPostfix, (int) (rho * 100 + 0.1),
	    (int) (tau * 1000),
	    (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), nTraj);
    sprintf(postfix, "%s", dstGridPostfix);
    sprintf(forwardTransitionFileName,
	    "%s/transfer/forwardTransition/forwardTransition%s.h5",
	    resDir, postfix);

    // Create HDF5 file
    EpetraExt::HDF5 HDF5(Comm);
    HDF5.Create(forwardTransitionFileName);
    
    // Set random number generator
    gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);
    // Get seed and set random number generator
    gsl_rng_set(r, Comm.MyPID()+1);
  
    // Define grid and allocate grid membership matrix
    Grid *grid = new RegularGrid(nx, gridLimitsLow, gridLimitsUp);
    // Print grid
    if (MyPID == 0)
      grid->printGrid(gridFileName, "%.12lf", true);
    
    // Construct a Map that puts approximately the same number of
    // equations on each process.
    if (MyPID == 0)
      std::cout << "Constructing a map for " << N << " elements..."
		<< std::endl;
    Epetra_Map rowMap (N, 0, Comm);
    
    // Storage for this processor's nonzeros.
    const int localsize = nTrajPerBox;
    int *jv = (int *) malloc(localsize * sizeof(int));
    double *vv = (double *) malloc(localsize * sizeof(double));
    
    // Initial condition and final box membership per initial box
    gsl_vector *IC = gsl_vector_alloc(dim);
    gsl_vector_uint *multiIdx = gsl_vector_uint_alloc(dim);
    gsl_vector *minBox = gsl_vector_alloc(dim);
    gsl_vector *maxBox = gsl_vector_alloc(dim);
    
    // Define field
    vectorField *field = new Lorenz63(rho, sigma, beta);
    // Define numerical scheme
    numericalScheme *scheme = new RungeKutta4(dim);
    // Define model (the initial state will be assigned later)
    model *mod = new model(field, scheme);


    /**
     * BUILD TRANSITION MATRIX
     */
    Epetra_Time timer(Comm);

    // Construct with StaticProfile=true since we know numNonzerosPerRow.
    // Less memory will be needed in FillComplete.
    P = new Epetra_CrsMatrix(Copy, *rowMap, nTrajPerBox, true);
    
    // Srinkle box by box
    if (MyPID == 0)
      std::cout << "\nConstructing transfer matrix for a lag of "
		<< tau << std::endl;

    nIn = 0;
    for (size_t box0 = 0; box0 < N; box0++) {
      if (rowMap->MyGID(box0)) {
	// Verbose
	if (box0 % (N / 100) == 0)
	  std::cout << "Getting transitions from box " << box0
		    << " of " << N-1 << std::endl;
	
	// Get boundaries of box
	getBoxBoundaries(box0, nx, multiIdx, grid, minBox, maxBox);

	// Simulate trajecories from uniformly sampled initial cond in box
	traj = 0;
	while (traj < (size_t) nTrajPerBox) {
	  
	  // Get random initial distribution
	  for (size_t d = 0; d < (size_t) dim; d++)
	    gsl_vector_set(IC, d, gsl_ran_flat(r, gsl_vector_get(minBox, d),
					       gsl_vector_get(maxBox, d)));
	  
	  // Get trajectory
	  if ((boxf = getTraj(mod, grid, IC, tau, dt)) < N) {
	    jv[traj] = boxf;
	    vv[traj] = 1. / nTrajPerBox;
	    traj++;
	  }
	  nTot++;
	}
	nIn += traj;
	  
	// Save transitions of box0
	if (P->InsertGlobalValues(box0, traj, vv, jv) < 0)
	  EPETRA_CHK_ERR(ierr);
      }
    }
    std::cout <<  (nTot - nIn) * 100. / nTot
	      << "% of the trajectories ended up outside the domain for "
	      << MyPID << std::endl;
    // Finalize matrix (sum duplicates, in particular)
    EPETRA_CHK_ERR(A->FillComplete());

    // Elapsed time
    double dt = timer.ElapsedTime();
    if (MyPID == 0)
      std::cout << "Transition matrix build time (secs):  " << dt << std::endl;

    free(jv);
    free(vv);
    gsl_vector_free(IC);
    gsl_vector_free(minBox);
    gsl_vector_free(maxBox);
    delete mod;
    delete scheme;
    delete field;
    delete grid;
  
    // Write forward transition matrix
    if (MyPID == 0)
      std::cout << "Writing forward transition matrix..."
		<< std::endl;
    HDF5.Write("forwardTransitionMatrix", *P);

#ifdef EPETRA_MPI
    MPI_Finalize () ;
#endif // EPETRA_MPI
  }

  // Free
  freeConfig();
  
  return 0;
}


/** 
 * Conversion from spherical to Cartesian coordinates
 * centered at (0, 0, rho + sigma) and of unit radius rho + sigma.
 * \param[inout] x Coordinates of vector to convert.
 */
void
sphere2Cart(gsl_vector *X)
{
  double x, y, z;
  double r, theta, phi;
  
  r = gsl_vector_get(X, 0);
  theta = gsl_vector_get(X, 1);    
  phi = gsl_vector_get(X, 2);

  r *= rho + sigma;
  
  x = r * sin(theta) * cos(phi);
  y = r * sin(theta) * sin(phi);
  z = r * cos(theta);

  gsl_vector_set(X, 0, x);
  gsl_vector_set(X, 1, y);
  gsl_vector_set(X, 2, z + rho + sigma);
  
  return;
}

/** 
 * Conversion from Cartesian to spherical coordinates
 * centered at (0, 0, rho + sigma) and of unit radius rho + sigma.
 * \param[inout] x Coordinates of vector to convert.
 */
void
cart2Sphere(gsl_vector *X)
{
  double r, theta, phi;
  double x, y, z;

  x = gsl_vector_get(X, 0);
  y = gsl_vector_get(X, 1);    
  z = gsl_vector_get(X, 2) - (rho + sigma);

  r = sqrt(gsl_pow_2(x) + gsl_pow_2(y) + gsl_pow_2(z));
  theta = acos(z / r);
  phi = atan2(y, x);
  r /= rho + sigma;


  gsl_vector_set(X, 0, r);
  gsl_vector_set(X, 1, theta);
  gsl_vector_set(X, 2, phi);
  
  return;
}


int
getTraj(model *mod, Grid *grid, const gsl_vector *IC, const double tau,
	const double dt)
{
  int boxf;
  
  // Convert initial condition from spherical
  // to Cartesian coordinates
  sphere2Cart(IC);

  // Numerical integration
  mod->integrate(IC, tau, dt);

  // Convert final state from Cartesian
  // to adapted spherical coordinates
  cart2Sphere(mod->current);
  
  // Get box of final state
  boxf = grid->getBoxMembership(mod->current);
  
  return boxf;
}


void
getBoxBoundaries(size_t box0, gsl_vector_uint *nx, gsl_vector_uint *multiIdx,
		 Grid *grid, gsl_vector *minBox, gsl_vector *maxBox)
{
  size_t dim = nx->size;
  
  unravel_index(box0, nx, multiIdx);
  for (size_t d = 0; d < dim; d++) {
    // Get lower limit of box0 for dim d from grid bounds
    gsl_vector_set(minBox, d,
		   gsl_vector_get(grid->bounds->at(d),
				  gsl_vector_uint_get(multiIdx, d)));
    
    // Get upper limit of box0 for dim d from grid bounds
    gsl_vector_set(maxBox, d,
		   gsl_vector_get(grid->bounds->at(d),
				  gsl_vector_uint_get(multiIdx, d) + 1));
  }
  
  return;
}
