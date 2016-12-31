#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <transferOperator.hpp>
#include <transferSpectrum.hpp>
#include "../cfg/readConfig.hpp"


/** \file spectrum.cpp
 *  \ingroup examples
 *  \brief Get spectrum of transfer operators.
 *   
 *  Get spectrum of transfer operators.
 */


/** \brief Calculate the spectrum of a transfer operator.
 * 
 * After parsing the configuration file,
 * the transition matrices are then read from matrix files in coordinate format.
 * The Eigen problem is then defined and solved using ARPACK++.
 * Finally, the results are written to file.
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
      readSpectrum(&cfg);
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
  
  // Declarations
  // Transfer
  char forwardTransitionFileName[256], srcPostfix[256], srcPostfixSim[256],
    postfix[256];
  transferOperator *transferOp;

  // Eigen problem
  char EigValForwardFileName[256];
  transferSpectrum *transferSpec;

  
  // Scan matrices and distributions for one lag
  const double tau = gsl_vector_get(tauRng, 0);
  std::cout << "\nGetting spectrum for a lag of " << tau << std::endl;

  // Get file names
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(srcPostfixSim, "%s%s_rho%04d_L%d_dt%d_nTraj%d",
	  srcPostfix, gridPostfix, (int) (rho * 100 + 0.1), (int) (tau * 1000),
	  (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), nTraj);
  sprintf(postfix, "%s", srcPostfixSim);
  sprintf(forwardTransitionFileName, \
	  "%s/transfer/forwardTransition/forwardTransition%s.coo%s",
	  resDir, postfix, fileFormat);
  sprintf(EigValForwardFileName,
	  "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);
  sprintf(EigValForwardFileName,
	  "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);

  // Read transfer operator
  std::cout << "Reading stationary transfer operator..." << std::endl;
  try
    {
      /** Construct transfer operator allocating memory
       *  to initial distribution and mask (set uniform). */
      transferOp = new transferOperator(N, stationary);
      
      /** Set uniform initial distribution (for eigenvector norm). */
      gsl_vector_set_all(transferOp->initDist, 1. / N);
	    
      // Scan forward transition matrix (this sets NFilled)
      std::cout << "Scanning forward transition matrix from "
		<< forwardTransitionFileName << std::endl;
      transferOp->scanForwardTransition(forwardTransitionFileName,
					fileFormat);
    }
  catch (std::exception &ex)
    {
      std::cerr << "Error reading transfer operator: " << ex.what()
		<< std::endl;
      return EXIT_FAILURE;
    }

      
  // Get spectrum
  try
    {
      // Solve eigen value problem with default configuration
      transferSpec = new transferSpectrum(nev, transferOp, config);

      std::cout << "Solving eigen problem for forward transition matrix..."
		<< std::endl;
      transferSpec->getSpectrumForward();
      std::cout << "Found "
		<< transferSpec->getNev()
		<< "/" << nev << " eigenvalues." << std::endl;
    }
  catch (std::exception &ex)
    {
      std::cerr << "Error calculating spectrum: " << ex.what() << std::endl;
      return EXIT_FAILURE;
    }
  
  // Write spectrum 
  try
    {
      std::cout << "Writing forward eigenvalues and eigenvectors..."
		<< std::endl;
      transferSpec->writeEigValForward(EigValForwardFileName,
				       fileFormat);
    }
  catch (std::exception &ex)
    {
      std::cerr << "Error writing spectrum: " << ex.what() << std::endl;
      return EXIT_FAILURE;
    }

  // Free
  delete transferSpec;
  delete transferOp;
  freeConfig();
  
return 0;
}
