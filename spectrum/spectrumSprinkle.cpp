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
  catch(const SettingNotFoundException &nfex) {
    std::cerr << "Setting " << nfex.getPath() << " not found." << std::endl;
    throw nfex;
  }
  catch(const FileIOException &fioex) {
    std::cerr << "I/O error while reading configuration file." << std::endl;
    throw fioex;
  }
  catch(const ParseException &pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    throw pex;
  }
  catch(const SettingTypeException &stex) {
    std::cerr << "Setting type exception." << std::endl;
    throw stex;
  }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }
  
  // Declarations
  // Transfer
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256],
    maskFileName[256], srcPostfix[256], srcPostfixSim[256], postfix[256];
  transferOperator *transferOp;
  gsl_vector *initDist, *finalDist;
  gsl_vector_uint *mask;

  // Eigen problem
  char EigValForwardFileName[256], EigVecForwardFileName[256],
    EigValBackwardFileName[256], EigVecBackwardFileName[256];
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
  sprintf(backwardTransitionFileName, \
	  "%s/transfer/backwardTransition/backwardTransition%s.coo%s",
	  resDir, postfix, fileFormat);
  sprintf(EigValForwardFileName,
	  "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);
  sprintf(EigValForwardFileName,
	  "%s/spectrum/eigval/eigvalForward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);
  sprintf(EigVecForwardFileName,
	  "%s/spectrum/eigvec/eigvecForward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);
  sprintf(EigValBackwardFileName,
	  "%s/spectrum/eigval/eigvalBackward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);
  sprintf(EigVecBackwardFileName,
	  "%s/spectrum/eigvec/eigvecBackward_nev%d%s.%s",
	  resDir, nev, postfix, fileFormat);

  // Read transfer operator
  std::cout << "Reading stationary transfer operator..." << std::endl;
  try
    {
      /** Construct transfer operator without allocating memory
	  to the distributions (only to the mask) ! */
      transferOp = new transferOperator(N, stationary);
	    
      // Scan forward transition matrix (this sets NFilled)
      std::cout << "Scanning forward transition matrix from "
		<< forwardTransitionFileName << std::endl;
      transferOp->scanForwardTransition(forwardTransitionFileName,
					fileFormat);

      // Scan mask for the first lag
      sprintf(maskFileName, "%s/transfer/mask/mask%s.%s",
	      resDir, postfix, fileFormat);
      std::cout << "Scanning mask from "
		<< maskFileName << std::endl;
      transferOp->scanMask(maskFileName, fileFormat);
      
      // Save mask
      mask = gsl_vector_uint_alloc(transferOp->getN());
      gsl_vector_uint_memcpy(mask, transferOp->mask);

      // Allocate memory to distributions
      transferOp->allocateDist();

      // Scan initial distribution for the first lag
      sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.%s",
	      resDir, postfix, fileFormat);
      std::cout << "Scanning initial distribution from "
		<< initDistFileName << std::endl;
      transferOp->scanInitDist(initDistFileName,
			       fileFormat);
      
      // Save initial distribution
      initDist = gsl_vector_alloc(transferOp->getNFilled());
      gsl_vector_memcpy(initDist, transferOp->initDist);

      if (!stationary)
	{
	  // Scan backward transition matrix
	  std::cout << "Scanning backward transition matrix from "
		    << backwardTransitionFileName << std::endl;
	  transferOp->scanBackwardTransition(backwardTransitionFileName,
					     fileFormat);
      
	  // Only scan final distribution for the first lag
	  sprintf(finalDistFileName,
		  "%s/transfer/finalDist/finalDist%s.%s",
		  resDir, postfix, fileFormat);
	  std::cout << "Scanning final distribution from "
		    << finalDistFileName << std::endl;
	  transferOp->scanFinalDist(finalDistFileName,
				    fileFormat);
      
	  // Save final distribution
	  finalDist = gsl_vector_alloc(transferOp->getNFilled());
	  gsl_vector_memcpy(finalDist, transferOp->finalDist);
	}
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

      if (getForwardEigenvectors)
	{
	  std::cout << "Solving eigen problem for forward transition matrix..."
		    << std::endl;
	  transferSpec->getSpectrumForward();
	  std::cout << "Found "
		    << transferSpec->getNev()
		    << "/" << nev << " eigenvalues." << std::endl;
	}
      if (getBackwardEigenvectors)
	{
	  std::cout << "Solving eigen problem for backward transition matrix..."
		    << std::endl;
	  transferSpec->getSpectrumBackward();
	  std::cout << "Found "
		    << transferSpec->getNev()
		    << "/" << nev << " eigenvalues." << std::endl;
	}
      if (getForwardEigenvectors
	  && getBackwardEigenvectors
	  && makeBiorthonormal)
	{
	  std::cout << "Making set of forward and backward eigenvectors \
biorthonormal..."
		    << std::endl;
	  transferSpec->makeBiorthonormal();
	}
    }
  catch (std::exception &ex)
    {
      std::cerr << "Error calculating spectrum: " << ex.what() << std::endl;
      return EXIT_FAILURE;
    }
  
  // Write spectrum 
  try
    {
      if (getForwardEigenvectors)
	{
	  std::cout << "Writing forward eigenvalues and eigenvectors..."
		    << std::endl;
	  // transferSpec->writeSpectrumForward(EigValForwardFileName,
	  // 				     EigVecForwardFileName,
	  // 				     fileFormat);
	  transferSpec->writeEigValForward(EigValForwardFileName,
					   fileFormat);
	}
      if (getBackwardEigenvectors)
	{
	  std::cout << "Writing backward eigenvalues and eigenvectors..."
		    << std::endl;
	  // transferSpec->writeSpectrumBackward(EigValBackwardFileName,
	  // 				      EigVecBackwardFileName,
	  // 				      fileFormat);
	  transferSpec->writeEigValBackward(EigValBackwardFileName,
					    fileFormat);
	}
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
  if (initDist)
    gsl_vector_free(initDist);
  if (finalDist)
    gsl_vector_free(finalDist);
  
return 0;
}
