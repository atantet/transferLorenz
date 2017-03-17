#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>

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
#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_MultiVectorOut.h>
// Include header to provide Anasazi with Epetra adapters.  If you
// plan to use Tpetra objects instead of Epetra objects, include
// AnasaziTpetraAdapter.hpp instead; do analogously if you plan to use
// Thyra objects instead of Epetra objects.
#include <AnasaziEpetraAdapter.hpp>
// Include header to define eigenproblem Ax = \lambda*x
#include <AnasaziBasicEigenproblem.hpp>
// Include header for block Davidson eigensolver
#include <AnasaziBlockKrylovSchurSolMgr.hpp>
#include <AnasaziBasicOutputManager.hpp>
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_math.h>
#include <gsl_extension.hpp>

#include "../cfg/readConfig.hpp"

/** \file spectrumAnasazi.cpp
 *  \ingroup examples
 *  \brief Get spectrum of transfer operators using Anasazi.
 *   
 *  Get spectrum of transfer operators.
 */


int readBinaryCrsMatrix(const char *filename, const Epetra_Comm & Comm,
			const Epetra_Map &rowMap, Epetra_CrsMatrix *&A,
			const bool transpose);

/** \brief Calculate the spectrum of a transfer operator.
 * 
 * After parsing the configuration file,
 * the transition matrices are then read from matrix files
 * in coordinate format.
 * The Eigen problem is then defined and solved using ARPACK++.
 * Finally, the results are written to file.
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
      readSpectrum(&cfg);
      std::cout << "Sparsing success.\n" << std::endl;
    }
  catch(const SettingTypeException &ex) {
    std::cerr << "Setting " << ex.getPath() << " type exception."
	      << std::endl;
    throw ex;
  }
  catch(const SettingNotFoundException &ex) {
    std::cerr << "Setting " << ex.getPath() << " not found." << std::endl;
    throw ex;
  }
  catch(const SettingNameException &ex) {
    std::cerr << "Setting " << ex.getPath() << " name exception."
	      << std::endl;
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
  catch (...) {
    std::cerr << "Error reading configuration file" << std::endl;
    return(EXIT_FAILURE);
  }

  // Declarations
  // Transfer
  char forwardTransitionFileName[256], srcPostfix[256],
    srcPostfixSim[256], postfix[256];

  // Eigen problem
  char EigValForwardFileName[256];
  char EigVecForwardFileName[256];

  // Eigen solver configuration
  std::string which ("LM");
  int blockSize = 1;
  int numBlocks = nev * 6;
  int numRestarts = 100;
  // Size of matrix nx*nx
  const int NumGlobalElements = N;

  // Scan matrices and distributions
  const double tau = gsl_vector_get(tauRng, 0);
  std::cout << "\nGetting spectrum for a lag of " << tau << std::endl;
  
  // Get file names
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(srcPostfixSim, "%s%s_rho%04d_L%d_dt%d_nTraj%d", srcPostfix,
	  gridPostfix, (int) (rho * 100 + 0.1), (int) (tau * 1000),
	  (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), nTraj);
  sprintf(postfix, "%s", srcPostfixSim);
  sprintf(forwardTransitionFileName, \
	  "%s/transfer/forwardTransition/forwardTransition%s.coo%s",
	  resDir, postfix, fileFormat);
  sprintf(EigValForwardFileName,
	  "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);
  sprintf(EigVecForwardFileName,
	  "%s/spectrum/eigval/eigvecBackward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);

  bool success = false;
  try {
    // Create an Anasazi output manager
    Anasazi::BasicOutputManager<ScalarType> printer;

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

    
    // Construct a Map that puts approximately the same number of
    // equations on each process.
    if (MyPID == 0)
      std::cout << "Constructing a map for " << NumGlobalElements
		<< " elements..." << std::endl;
    Epetra_Map Map (NumGlobalElements, 0, Comm);
  
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


    /*
     * Read forward transition matrix
     */
    std::cout << "Reading transition matrix and transpose..."
	      << std::endl;
    Epetra_CrsMatrix *PT;
    if (readBinaryCrsMatrix(forwardTransitionFileName, comm,
			    Map, PT, true)) {
      if (MyPID = 0) {
	printer.print (Anasazi::Errors, "readBinaryCrsMatrix::setProblem() \
reported an error.\n");
      }
#ifdef EPETRA_MPI
      MPI_Finalize ();
#endif // EPETRA_MPI
      throw -1;
    }
    std::cout << "=============================================\
===================================" << std::endl;
      
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcpFromRef(*PT);

    // Create an Epetra_MultiVector for an initial vector to start the
    // solver.  Note: This needs to have the same number of columns as
    // the blocksize.
    Teuchos::RCP<Epetra_MultiVector> ivec
      = Teuchos::rcp (new Epetra_MultiVector (Map, blockSize));
    ivec->Random (); // fill the initial vector with random values


    /*
     * Create the eigenproblem.
     */
    std::cout << "Setting eigen problem..." << std::endl;
    Teuchos::RCP<Anasazi::BasicEigenproblem<ScalarType, MV, OP> >
      MyProblem = Teuchos::rcp
      (new Anasazi::BasicEigenproblem<ScalarType, MV, OP> (A, ivec));

    // Set the number of eigenvalues requested
    MyProblem->setNEV (nev);

    // Inform the eigenproblem that you are finishing passing it information
    bool successProb = MyProblem->setProblem ();
    if (! successProb) {
      if (MyPID = 0) {
	printer.print (Anasazi::Errors, "Anasazi::BasicEigenproblem\
::setProblem() reported an error.\n");
      }
#ifdef EPETRA_MPI
      MPI_Finalize ();
#endif // EPETRA_MPI
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
    std::cout << "Creating eigen solver..." << std::endl;
    Teuchos::RCP<Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP> >
      MySolverMan = Teuchos::rcp
      (new Anasazi::BlockKrylovSchurSolMgr<ScalarType, MV, OP>
       (MyProblem, MyPL));

    
    /*
     * Solve the problem
     */
    std::cout << "Solving eigen problem..." << std::endl;
    Anasazi::ReturnType returnCode = MySolverMan->solve ();

    // Get the eigenvalues and eigenvectors from the eigenproblem
    Anasazi::Eigensolution<ScalarType, MV> sol = MyProblem->getSolution ();
    std::vector<Anasazi::Value<ScalarType> > evals = sol.Evals;
    std::vector<int> index = sol.index;
    int numev = sol.numVecs;
    std::cout << "Found " << numev << " eigenvalues" << std::endl;


    /*
     * Print eigenvalues and indices
     */
    // Print the results on MPI process 0.
    if (MyPID == 0) {
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
  }

  // Free
  freeConfig();
  
#ifdef EPETRA_MPI
  MPI_Finalize () ;
#endif // EPETRA_MPI

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}


int
readBinaryCrsMatrix(const char *filename,
		    const Epetra_Comm &Comm,
		    const Epetra_Map &rowMap,
		    Epetra_CrsMatrix *&A,
		    const bool transpose)
{
  int *indices, *NumEntriesPerRow;
  const int MyPID = Comm.MyPID ();
  
  if (MyPID == 0) {
    gsl_spmatrix *m;
    // Open stream
    FILE * handle = 0;
    std::cout << "Opening binary CRS file "
	      << filename << std::endl;
    // Open file
    if (!(handle = fopen(filename,"r")))
      EPETRA_CHK_ERR(-1); // file not found
    
    // Read COO matrix (if first)
    std::cout << "Reading file" << std::endl;
    if (transpose)
      m = gsl_spmatrix_alloc2read(handle, GSL_SPMATRIX_CCS);
    else
      gsl_spmatrix *m = gsl_spmatrix_alloc2read(handle, GSL_SPMATRIX_CRS);
    gsl_spmatrix_fread(handle, m);
    if (!m)
      EPETRA_CHK_ERR(-1); // file not found
    if (transpose)
      gsl_spmatrix_transpose2(m);
    M = m->size1;
    N = m->size2;
    NZ = m->nz;
  }
  comm.Broadcast(&M, 1, 0);
  comm.Broadcast(&N, 1, 0);
  comm.Broadcast(&NZ, 1, 0);

  // Get number of entries per row
  NumEntriesPerRow = (int *) malloc(m->size1 * sizeof(int));
  for (int i = 0; i < (int) m->size1; i++)
    NumEntriesPerRow[i] = (int) (m->p[i+1] - m->p[i]);

  // Create matrix
  A = new Epetra_CrsMatrix(Copy, rowMap, NumEntriesPerRow, true);

  // Insert each row
  std::cout << "Inserting rows" << std::endl;
  for (int i = 0; i < (int) m->size1; i++) {
    NumEntriesPerRow[i] = (int) (m->p[i+1] - m->p[i]);
    if (NumEntriesPerRow[i])
      {
	// Convert indices from size_t to int
	indices = (int *) malloc(NumEntriesPerRow[i] * sizeof(int));
	for (int k = 0; k < NumEntriesPerRow[i]; k++)
	  indices[k] = m->i[m->p[i] + k];
	// Insert row
	int ierr = A->InsertGlobalValues(i, NumEntriesPerRow[i],
					 &(m->data[m->p[i]]), indices);
	if (ierr<0)
	  EPETRA_CHK_ERR(ierr);
	// Free indices
	free(indices);
      }
  }
  // Free number of entries per row
  free(NumEntriesPerRow);

  // Complete
  EPETRA_CHK_ERR(A->FillComplete());    

  return 0;
}
