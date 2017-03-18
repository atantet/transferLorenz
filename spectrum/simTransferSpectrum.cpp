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
#include <EpetraExt_ConfigDefs.h>
#ifdef HAVE_MPI
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
#include <Epetra_Time.h>
#include <EpetraExt_Exception.h>
#include <EpetraExt_MultiVectorOut.h>
#include <AnasaziEpetraAdapter.hpp>
#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziBlockKrylovSchurSolMgr.hpp>
#include <AnasaziBasicOutputManager.hpp>
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_StandardCatchMacros.hpp"


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
long long getTraj(model *mod, Grid *grid, gsl_vector *IC, const double tau,
	    const double dt);
/** \brief Get box boundaries. */
void getBoxBoundaries(const size_t box0, const gsl_vector_uint *nx,
		      const Grid *grid, gsl_vector *minBox, gsl_vector *maxBox);


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
  // Type definitions
  typedef double ScalarType;
  typedef Teuchos::ScalarTraits<ScalarType>          SCT;
  typedef SCT::magnitudeType               MagnitudeType;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Anasazi::MultiVecTraits<ScalarType, MV> MVT;
  
#ifdef HAVE_MPI
  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Create an Epetra communicator
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  const int MyPID = Comm.MyPID ();
  const int NumProc = Comm.NumProc ();
    
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
      bool verboseCFG;
      if (MyPID == 0) {
	std::cout << "Sparsing config file " << configFileName << std::endl;
	verboseCFG = true;
      }
      else
	verboseCFG = false;
      cfg.readFile(configFileName);
      readGeneral(&cfg, verboseCFG);
      readModel(&cfg, verboseCFG);
      readSimulation(&cfg, verboseCFG);
      readSprinkle(&cfg, verboseCFG);
      readGrid(&cfg, verboseCFG);
      readTransfer(&cfg, verboseCFG);
      readSpectrum(&cfg, verboseCFG);
      if (MyPID == 0)
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

  bool success = false;
  try {
    // Create an Anasazi output manager
    Anasazi::BasicOutputManager<ScalarType> printer;

    // Scale the number of trajectories with the number of processors
    const long long NumGlobalElements = (long long) N;
    const long long nTrajPerProc = (long long) nTraj;
    const long long nTrajPerBox = nTrajPerProc / NumGlobalElements * NumProc;
    long long nIn = 0;
    long long nTot = 0;

    // Transition step
    const double tau = gsl_vector_get(tauRng, 0);
    
    // Eigen problem
    char EigValForwardFileName[256], EigVecForwardFileName[256];

    // Eigen solver configuration
    std::string which ("LM");
    int blockSize = 1;
    int numBlocks = nev * 6;
    int numRestarts = 100;

    // Transfer operator declarations
    char postfix[256];
    char srcPostfix[256], dstGridPostfix[256], gridFileName[256];
    sprintf(srcPostfix, "_%s", caseName);
    sprintf(gridFileName, "%s/grid/grid%s%s.txt", resDir, srcPostfix,
	    gridPostfix);
    sprintf(dstGridPostfix, "%s%s_rho%04d_L%d_dt%d_nTraj%d_nProc%d",
	    srcPostfix, gridPostfix, (int) (rho * 100 + 0.1),
	    (int) (tau * 1000),
	    (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), nTraj, NumProc);
    sprintf(postfix, "%s", dstGridPostfix);
    sprintf(EigValForwardFileName,
	    "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	    resDir, nev, postfix, fileFormat);
    sprintf(EigVecForwardFileName,
	    "%s/spectrum/eigval/eigvecBackward_nev%d%s.%s",
	    resDir, nev, postfix, fileFormat);

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
      std::cout << "Constructing a map for " << NumGlobalElements
		<< " elements..." << std::endl;
    Epetra_Map rowMap (NumGlobalElements, 0, Comm);
    
    // Create a sort manager to pass into the block Krylov-Schur
    // solver manager
    // -->  Make sure the reference-counted pointer is of type
    // Anasazi::SortManager<>
    // -->  The block Krylov-Schur solver manager uses
    // Anasazi::BasicSort<> by default,
    //      so you can also pass in the parameter "Which",
    // instead of a sort manager.
    Teuchos::RCP<Anasazi::SortManager<MagnitudeType> > MySort =
      Teuchos::rcp( new Anasazi::BasicSort<MagnitudeType>( which ) );

    // Epetra timer
    Epetra_Time timer(Comm);

    // Construct with StaticProfile=true since we know numNonzerosPerRow.
    // Less memory will be needed in FillComplete.
    Epetra_CrsMatrix *P
      = new Epetra_CrsMatrix(Copy, rowMap, nTrajPerBox, true);
    
    if (MyPID == 0)
      std::cout << "\nConstructing transition matrix for a lag of "
		<< tau << " from " << nTrajPerProc
		<< " trajectories for each process out of "
		<< NumProc << std::endl;

#pragma omp parallel
    {
      // Storage for this processor's nonzeros.
      long long *jv = (long long *) malloc(nTrajPerBox * sizeof(long long));
      double *vv = (double *) malloc(nTrajPerBox * sizeof(double));
      long long boxf, traj, trajTot;
    
      // Initial condition and final box membership per initial box
      gsl_vector *IC = gsl_vector_alloc(dim);
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
      // Srinkle box by box
#pragma omp for
      for (long long box0 = 0; box0 < NumGlobalElements ; box0++) {
	if (rowMap.MyGID(box0)) {
	  // Verbose
	  if (box0 % (NumGlobalElements / 100) == 0) {
#pragma omp critical
	    {
	      std::cout << "Getting transitions from box " << box0 << " of "
			<< NumGlobalElements-1 << " by " << MyPID << std::endl;
	    }
	  }
	
	  // Get boundaries of box
	  getBoxBoundaries((size_t) box0, nx, grid, minBox, maxBox);

	  // Simulate trajecories from uniformly sampled initial cond in box
	  traj = 0;
	  trajTot = 0;
	  while (traj < nTrajPerBox) {
	    // Get random initial distribution
	    for (size_t d = 0; d < (size_t) dim; d++)
	      gsl_vector_set(IC, d, gsl_ran_flat(r, gsl_vector_get(minBox, d),
						 gsl_vector_get(maxBox, d)));
	  
	    // Get trajectory
	    if ((boxf = getTraj(mod, grid, IC, tau, dt)) < NumGlobalElements) {
	      jv[traj] = boxf;
	      vv[traj] = 1. / nTrajPerBox;
	      traj++;
	    }
	    trajTot++;
	  }
	  
	  // Save transitions of box0
	  int ierr;
#pragma omp critical
	  {
	    if ((ierr = P->InsertGlobalValues(box0, traj, vv, jv)) < 0) {
	      std::cerr << "Error: inserting global values to transition matrix."
			<< std::endl;
	      throw std::exception();
	    }
	    nIn += traj;
	    nTot += trajTot;
	  }
	}
      }
      free(jv);
      free(vv);
      gsl_vector_free(IC);
      gsl_vector_free(minBox);
      gsl_vector_free(maxBox);
      delete mod;
      delete scheme;
      delete field;
    }
    std::cout <<  (nTot - nIn) * 100. / nTot
	      << "% of the trajectories ended up outside the domain for "
	      << MyPID << std::endl;
    
    // Finalize matrix (sum duplicates, in particular)
    EPETRA_CHK_ERR(P->FillComplete());

    // Elapsed time
    double dt = timer.ElapsedTime();
    if (MyPID == 0)
      std::cout << "Transition matrix build time (secs):  " << dt << std::endl;
    delete grid;



    /**
     * SOLVE EIGEN PROBLEM
     */
    timer.ResetStartTime();
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcpFromRef(*P);

    // Create an Epetra_MultiVector for an initial vector to start the
    // solver.  Note: This needs to have the same number of columns as
    // the blocksize.
    Teuchos::RCP<Epetra_MultiVector> ivec
      = Teuchos::rcp (new Epetra_MultiVector (rowMap, blockSize));
    ivec->Random (); // fill the initial vector with random values


    /** Create the eigenproblem. */
    if (MyPID == 0)
      std::cout << "Setting eigen problem..." << std::endl;
    Teuchos::RCP<Anasazi::BasicEigenproblem<ScalarType, MV, OP> >
      MyProblem = Teuchos::rcp
      (new Anasazi::BasicEigenproblem<ScalarType, MV, OP> (A, ivec));

    // Set the number of eigenvalues requested
    MyProblem->setNEV (nev);

    // Inform the eigenproblem that you are finishing passing it information
    bool successProb = MyProblem->setProblem ();
    if (! successProb) {
      printer.print (Anasazi::Errors, "Anasazi::BasicEigenproblem\
::setProblem() reported an error.\n");
      throw -1;
    }

    // Create parameter list to pass into the solver manager
    Teuchos::ParameterList MyPL;
    MyPL.set( "Sort Manager", MySort );
    MyPL.set ("Block Size", blockSize);
    MyPL.set( "Num Blocks", numBlocks );
    MyPL.set ("Maximum Iterations", config.maxit);
    MyPL.set( "Maximum Restarts", numRestarts);
    MyPL.set ("Convergence Tolerance", config.tol);
    MyPL.set ("Orthogonalization", "TSQR");
    MyPL.set ("Verbosity", Anasazi::Errors + Anasazi::Warnings \
	      + Anasazi::FinalSummary);

    // Create the solver manager
    if (MyPID == 0)
      std::cout << "Creating eigen solver..." << std::endl;
    Teuchos::RCP<Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP> >
      MySolverMan = Teuchos::rcp
      (new Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP>
       (MyProblem, MyPL));

    
    /*
     * Solve the problem
     */
    if (MyPID == 0)
      std::cout << "Solving eigen problem..." << std::endl;
    Anasazi::ReturnType returnCode = MySolverMan->solve ();

    // Get the eigenvalues and eigenvectors from the eigenproblem
    if (MyPID == 0)
      std::cout << "Getting solution..." << std::endl;
    Anasazi::Eigensolution<ScalarType, MV> sol = MyProblem->getSolution ();
    std::vector<Anasazi::Value<ScalarType> > evals = sol.Evals;
    std::vector<int> index = sol.index;
    int numev = sol.numVecs;
    if (MyPID == 0)
      std::cout << "Found " << numev << " eigenvalues" << std::endl;

    // Elapsed time
    dt = timer.ElapsedTime();
    if (MyPID == 0)
      std::cout << "Eigenproblem solved in time (secs):  " << dt << std::endl;
    
    /*
     * Print eigenvalues and indices
     */
    std::filebuf fb;
    std::ostream os(&fb);
    if (numev > 0)
      {
	fb.open (EigValForwardFileName, std::ios::out);
	for (int ev = 0; ev < numev; ev++)
	  os << evals[ev].realpart << "\t" << evals[ev].imagpart
	     << "\t" << index[ev] << std::endl;
	fb.close();
	  
	// Print eigenvectors
	if (getForwardEigenvectors)
	  {
	    Teuchos::RCP<MV> evecs = sol.Evecs;
	    EpetraExt::MultiVectorToMatrixMarketFile(EigVecForwardFileName,
						     *evecs);
	  }
      }
    success = true;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

#ifdef HAVE_MPI
  MPI_Finalize () ;
#endif // HAVE_MPI

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


long long
getTraj(model *mod, Grid *grid, gsl_vector *IC, const double tau,
	const double dt)
{
  long long boxf;
  
  // Convert initial condition from spherical
  // to Cartesian coordinates
  sphere2Cart(IC);

  // Numerical integration
  mod->integrate(IC, tau, dt);

  // Convert final state from Cartesian
  // to adapted spherical coordinates
  cart2Sphere(mod->current);
  
  // Get box of final state
  boxf = (long long) grid->getBoxMembership(mod->current);
  
  return boxf;
}


void
getBoxBoundaries(const size_t box0, const gsl_vector_uint *nx, const Grid *grid,
		 gsl_vector *minBox, gsl_vector *maxBox)
{
  size_t dim = nx->size;
  gsl_vector_uint *multiIdx = gsl_vector_uint_alloc(dim);
  
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
  // Free
  gsl_vector_uint_free(multiIdx);
  
  return;
}
