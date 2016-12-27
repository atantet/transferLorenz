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
  char srcFileName[256];
  FILE *srcStream;
  gsl_matrix *traj;
  size_t seed;
  double xmin, xmax;
  std::vector<gsl_matrix *> statesSeeds(nSeeds);

  // Grid declarations
  Grid *grid;

  // Grid membership declarations
  char gridMemFileName[256];
  FILE *gridMemStream;
  gsl_matrix_uint *gridMemMatrix;
  std::vector<gsl_vector_uint *> gridMemSeeds(nSeeds);
  bool getGridLimits = false;
    
  // Transfer operator declarations
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256],
    postfix[256], maskFileName[256];

  size_t tauNum;
  double tau;
  transferOperator *transferOp;

  
  // Get grid membership matrix
  if (! readGridMem)
    {
      // Allocate trajectory and grid limits
      traj = gsl_matrix_alloc(nt0, dim);
      if (!(gridLimitsLow && gridLimitsUp))
	{
	  getGridLimits = true;
	  gridLimitsLow = gsl_vector_alloc(dimObs);
	  gridLimitsUp = gsl_vector_alloc(dimObs);
	  gsl_vector_set_all(gridLimitsLow, 1.e30);
	  gsl_vector_set_all(gridLimitsUp, -1.e30);
	}
      
      // Iterate one simulation per seed
      for (size_t s = 0; s < nSeeds; s++)
	{
	  // Get seed and set random number generator
	  seed = gsl_vector_uint_get(seedRng, s);
	  
	  // Get membership vector
	  sprintf(srcFileName, "%s/simulation/sim%s_seed%d.%s",
		  resDir, srcPostfix, (int) seed, fileFormat);
	  
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
	  if (strcmp(fileFormat, "bin") == 0)
	    gsl_matrix_fread(srcStream, traj);
	  else
	    gsl_matrix_fscanf(srcStream, traj);

	  // Define observable
	  statesSeeds[s] = gsl_matrix_alloc(nt, (size_t) dimObs);
	  for (size_t d = 0; d < (size_t) dimObs; d++)
	    {
	      gsl_vector_const_view view 
		= gsl_matrix_const_subcolumn(traj,
					     gsl_vector_uint_get(components, d),
					     embedMax
					     - gsl_vector_uint_get(embedding, d),
					     nt);
	      gsl_matrix_set_col(statesSeeds[s], d, &view.vector);

	      // Save min max values of observable to define fix limits grid
	      if (getGridLimits)
		{
		  gsl_vector_minmax(&view.vector, &xmin, &xmax);
		  if (xmin < gsl_vector_get(gridLimitsLow, d))
		    gsl_vector_set(gridLimitsLow, d, xmin - 1.e-12);
		  if (xmax > gsl_vector_get(gridLimitsUp, d))
		    gsl_vector_set(gridLimitsUp, d, xmax + 1.e-12);
		}
	    }
	  
	  // Close trajectory file
	  fclose(srcStream);
	}

      // Free
      gsl_matrix_free(traj);

      
      // Define grid
      grid = new RegularGrid(nx, gridLimitsLow, gridLimitsUp);
    
      // Print grid
      grid->printGrid(gridFileName, "%.12lf", true);
    

      // Get grid membership for each seed
      for (size_t s = 0; s < nSeeds; s++)
	{
	  seed = gsl_vector_uint_get(seedRng, s);

	  // Grid membership file name
	  sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s_seed%d.%s",
		  resDir, gridPostfix, (int) seed, fileFormat);
  
	  // Open grid membership vector stream
	  if ((gridMemStream = fopen(gridMemFileName, "w")) == NULL)
	    {
	      fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	      perror("");
	      return(EXIT_FAILURE);
	    }
    
	  // Get grid membership vector
	  std::cout << "Getting grid membership vector for seed "
		    << seed << std::endl;
	  gridMemSeeds[s] = getGridMemVector(statesSeeds[s], grid);

	  // Write grid membership
	  if (strcmp(fileFormat, "bin") == 0)
	    gsl_vector_uint_fwrite(gridMemStream, gridMemSeeds[s]);
	  else
	    gsl_vector_uint_fprintf(gridMemStream, gridMemSeeds[s], "%d");

	  // Free states and close stream
	  gsl_matrix_free(statesSeeds[s]);
	  fclose(gridMemStream);
	}

      // Free
      delete grid;
    }
  else
    {
      // Read grid membership for each seed
      for (size_t s = 0; s < nSeeds; s++)
	{
	  seed = gsl_vector_uint_get(seedRng, s);

	  // Grid membership file name
	  sprintf(gridMemFileName, "%s/transfer/gridMem/gridMem%s_seed%d.%s",
		  resDir, gridPostfix, (int) seed, fileFormat);
	  
	  // Open grid membership stream for reading
	  std::cout << "Reading grid membership vector for seed "
		    << seed << " at " << gridMemFileName << std::endl;
	  
	  if ((gridMemStream = fopen(gridMemFileName, "r")) == NULL)
	    {
	      fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	      perror("");
	      return(EXIT_FAILURE);
	    }
      
	  // Read grid membership
	  gridMemSeeds[s] = gsl_vector_uint_alloc(nt);
	  if (strcmp(fileFormat, "bin") == 0)
	    gsl_vector_uint_fread(gridMemStream, gridMemSeeds[s]);
	  else
	    gsl_vector_uint_fscanf(gridMemStream, gridMemSeeds[s]);

	  // Close stream
	  fclose(gridMemStream);
	}
    }

  

  // Get transition matrices for the first lag only (memory reasons)
  size_t lag = 0;
  tau = gsl_vector_get(tauRng, lag);
  tauNum = (size_t) round(tau / printStep + 0.1);
  sprintf(postfix, "%s_tau%03d", gridPostfix, (int) (tau * 1000));

  std::cout << "\nConstructing transfer operator for a lag of "
	    << tau << std::endl;


  // Get full membership matrix
  std::cout << "Getting full membership matrix from the list \
of membership vecotrs..." << std::endl;
  gridMemMatrix = memVectorList2memMatrix(&gridMemSeeds, tauNum);

  // Free gridMemSeeds to save memory before to build transfer operator
  for (size_t s = 0; s < nSeeds; s++)
    gsl_vector_uint_free(gridMemSeeds[s]);

  
  // Get transition matrices as CSR
  std::cout << "Building stationary transfer operator..." << std::endl;
  transferOp = new transferOperator(gridMemMatrix, N, stationary);
  

  // Write results
  // Write forward transition matrix
  std::cout << "Writing forward transition matrix and initial distribution..."
	    << std::endl;
  sprintf(forwardTransitionFileName,
	  "%s/transfer/forwardTransition/forwardTransition%s.coo%s",
	  resDir, postfix, fileFormat);
  transferOp->printForwardTransition(forwardTransitionFileName,
				     fileFormat, "%.12lf");

  // Write mask and initial distribution
  sprintf(maskFileName, "%s/transfer/mask/mask%s.%s",
	  resDir, gridPostfix, fileFormat);
  transferOp->printMask(maskFileName,
			fileFormat, "%.12lf");
  
  sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.%s",
	  resDir, gridPostfix, fileFormat);
  transferOp->printInitDist(initDistFileName,
			    fileFormat, "%.12lf");
  
  // Write backward transition matrix
  if (!stationary)
    {
      std::cout << "Writing backward transition matrix \
and final distribution..." << std::endl;
      sprintf(backwardTransitionFileName,
	      "%s/transfer/backwardTransition/backwardTransition%s.coo%s",
	      resDir, postfix, fileFormat);
      transferOp->printBackwardTransition(backwardTransitionFileName,
					  fileFormat, "%.12lf");
      
      // Write final distribution 
      sprintf(finalDistFileName,
	      "%s/transfer/finalDist/finalDist%s.%s",
	      resDir, postfix, fileFormat);
      transferOp->printFinalDist(finalDistFileName,
				 fileFormat, "%.12lf");
    }

  
  // Free membership matrix and transfer operator
  delete transferOp;
  gsl_matrix_uint_free(gridMemMatrix);

  // Free config
  freeConfig();
  
  return 0;
}

