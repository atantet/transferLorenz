#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <ergoGrid.hpp>
#include <transferOperator.hpp>
#include <gsl_extension.hpp>
#include "../cfg/readConfig.hpp"


/** \file transfer.cpp
 *  \ingroup examples
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
 * The forward transition matrices as well as the initial distributions
 * are calculated from the membership matrix.
 * Note that, since the transitions are calculated from long time series,
 * the problem must be autonomous and ergodic (stationary) so that
 * the backward transition matrix and final distribution need not be calculated.
 * Finally, the results are printed.
 */


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
     readConfig(configFileName);
    }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  // Observable declarations
  FILE *srcStream;
  gsl_matrix *traj;
  gsl_matrix *states;

  // Grid declarations
  Grid *grid;

  // Grid membership declarations
  char gridMemFileName[256];
  FILE *gridMemStream;
  gsl_vector_uint *gridMemVector;
  gsl_matrix_uint *gridMemMatrix;
    
  // Transfer operator declarations
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256],
    postfix[256], maskFileName[256];

  size_t tauNum;
  double tau;
  transferOperator *transferOp;


  // Get membership vector
  sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s.%s",
	  resDir, gridPostfix, file_format);
  if (! readGridMem)
    {
      // Open time series file
      if ((srcStream = fopen(srcFileName, "r")) == NULL)
	{
	  fprintf(stderr, "Can't open source file %s for reading:",
		  srcFileName);
	  perror("");
	  return(EXIT_FAILURE);
	}

      // Read one-dimensional time series
      std::cout << "Reading trajectory in " << srcFileName << std::endl;
      traj = gsl_matrix_alloc(nt0, dim);
      if (strcmp(file_format, "bin") == 0)
	gsl_matrix_fread(srcStream, traj);
      else
	gsl_matrix_fscanf(srcStream, traj);

      // Close 
      fclose(srcStream);


      // Define observable
      states = gsl_matrix_alloc(nt, (size_t) dimObs);
      for (size_t d = 0; d < (size_t) dimObs; d++)
	{
	  gsl_vector_const_view view 
	    = gsl_matrix_const_subcolumn(traj,
					 gsl_vector_uint_get(components, d),
					 embedMax
					 - gsl_vector_uint_get(embedding, d),
					 nt);
	  gsl_matrix_set_col(states, d, &view.vector);
	}


      // Define grid
      grid = new RegularGrid(nx, nSTDLow, nSTDHigh, states);
    
      // Print grid
      grid->printGrid(gridFileName, "%.12lf", true);
    
      // Open grid membership vector stream
      if ((gridMemStream = fopen(gridMemFileName, "w")) == NULL){
	fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	perror("");
	return(EXIT_FAILURE);
      }
    
      // Get grid membership vector
      std::cout << "Getting grid membership vector" << std::endl;
      gridMemVector = getGridMemVector(states, grid);

      // Write grid membership
      if (strcmp(file_format, "bin") == 0)
	gsl_vector_uint_fwrite(gridMemStream, gridMemVector);
      else
	gsl_vector_uint_fprintf(gridMemStream, gridMemVector, "%d");
    
      // Close stream and free
      fclose(gridMemStream);
      gsl_matrix_free(traj);
      gsl_matrix_free(states);
      delete grid;
    }
  else
    {
      // Open grid membership stream for reading
      std::cout << "Reading grid membership vector at "
		<< gridMemFileName << std::endl;
      if ((gridMemStream = fopen(gridMemFileName, "r")) == NULL)
	{
	  fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	  perror("");
	  return(EXIT_FAILURE);
	}
      
      // Read grid membership
      gridMemVector = gsl_vector_uint_alloc(nt);
      if (strcmp(file_format, "bin") == 0)
	gsl_vector_uint_fread(gridMemStream, gridMemVector);
      else
	gsl_vector_uint_fscanf(gridMemStream, gridMemVector);

      // Close stream
      fclose(gridMemStream);
    }



  // Get transition matrices for different lags
  for (size_t lag = 0; lag < nLags; lag++)
    {
      tau = gsl_vector_get(tauRng, lag);
      tauNum = (size_t) round(tau / printStep + 0.1);
      sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));

      std::cout << "\nConstructing transfer operator for a lag of "
		<< tau << std::endl;


      // Get full membership matrix
      std::cout << "Getting full membership matrix from the list \
of membership vecotrs..." << std::endl;
      gridMemMatrix = memVector2memMatrix(gridMemVector, tauNum);

      
      // Get transition matrices as CSR
      std::cout << "Building stationary transfer operator..." << std::endl;
      transferOp = new transferOperator(gridMemMatrix, N, stationary);


      // Write results
      // Write forward transition matrix
      std::cout << "Writing forward transition matrix and initial distribution..."
		<< std::endl;
      sprintf(forwardTransitionFileName,
	      "%s/transfer/forwardTransition/forwardTransition%s.coo%s",
	      resDir, postfix, file_format);
      transferOp->printForwardTransition(forwardTransitionFileName,
					 file_format, "%.12lf");

      // Write mask and initial distribution
      if (lag == 0)
	{
	  sprintf(maskFileName, "%s/transfer/mask/mask%s.%s",
		  resDir, gridPostfix, file_format);
	  transferOp->printMask(maskFileName,
				file_format, "%.12lf");

	  sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.%s",
		  resDir, gridPostfix, file_format);
	  transferOp->printInitDist(initDistFileName,
				    file_format, "%.12lf");
	}
      
      // Write backward transition matrix
      if (!stationary)
	{
	  std::cout << "Writing backward transition matrix \
and final distribution..." << std::endl;
	  sprintf(backwardTransitionFileName,
		  "%s/transfer/backwardTransition/backwardTransition%s.coo%s",
		  resDir, postfix, file_format);
	  transferOp->printBackwardTransition(backwardTransitionFileName,
					      file_format, "%.12lf");

	  // Write final distribution 
	  if (lag == 0)
	    {
	      sprintf(finalDistFileName,
		      "%s/transfer/finalDist/finalDist%s.%s",
		      resDir, postfix, file_format);
	      transferOp->printFinalDist(finalDistFileName,
					 file_format, "%.12lf");
	    }
	}
	
      // Free
      delete transferOp;
      gsl_matrix_uint_free(gridMemMatrix);
    }

  // Free
  gsl_vector_uint_free(gridMemVector);
  freeConfig();
		
  return 0;
}

