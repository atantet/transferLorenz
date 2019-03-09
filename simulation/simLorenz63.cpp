#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simLorenz63.cpp 
 *  \ingroup examples
 *  \brief Simulate Lorenz (1963) deterministic flow.
 *
 *  Simulate Lorenz (1963) deterministic flow.
 */


/** \brief Simulation of Lorenz (1963) deterministic flow.
 *
 *  Simulation of Lorenz (1963) deterministic flow.
 *  After parsing the configuration file,
 *  the vector field of the Lorenz 1963 flow and the Runge-Kutta numerical scheme of order 4 are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  FILE *dstStream;
  gsl_matrix *X;
  char dstFileName[256], dstPostfix[256];
  size_t seed;

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

  // Set random number generator
  gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);

  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63(rho, sigma, beta);
  
  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim);

  // Define model (the initial state will be assigned later)
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Iterate one simulation per seed
  for (size_t s = 0; s < nSeeds; s++)
    {
      // Get seed and set random number generator
      seed = gsl_vector_uint_get(seedRng, s);
<<<<<<< HEAD
      printf("Setting random number generator with seed: %d\n", (int) seed);
      gsl_rng_set(r, seed);

      // Define names and open destination file
      sprintf(dstFileName, "%s/simulation/sim%s_seed%d.%s",
	      resDir, srcPostfix, (int) seed, fileFormat);
=======
      std::cout << "Setting random number generator with seed: " << seed
		<< std::endl;;
      gsl_rng_set(r, seed);

      // Define names and open destination file
      sprintf(dstPostfix, "%s_L%d_spinup%d_dt%d_samp%d", srcPostfixModel,
	      (int) L, (int) spinup,
	      (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	      (int) printStepNum);
      sprintf(dstFileName, "%s/simulation/sim%s_seed%d.%s",
	      resDir, dstPostfix, (int) seed, fileFormat);
>>>>>>> 753572aebb1cc5164212e76d5de8decc47e5d291
      if (!(dstStream = fopen(dstFileName, "w")))
	{
	  std::cerr << "Can't open " << dstFileName
		    << " for writing simulation: " << std::endl;;
	  perror("");
	  return EXIT_FAILURE;
	}

      // Get random initial distribution
      for (size_t d = 0; d < (size_t) dim; d++) {
	gsl_vector_set(initState, d,
		       gsl_ran_flat(r, gsl_vector_get(minInitState, d),
				    gsl_vector_get(maxInitState, d));

      // Set initial state
      printf("Setting initial state to (%.1lf, %.1lf, %.1lf)\n",
	     gsl_vector_get(initState, 0),
	     gsl_vector_get(initState, 1),
	     gsl_vector_get(initState, 2));
      mod->setCurrentState(initState);

      // Numerical integration
      std::cout << "Integrating simulation..." << std::endl;
      mod->integrate(initState, L, dt, spinup, printStepNum, &X);

      // Write results
      std::cout << "Writing..." << std::endl;
      if (strcmp(fileFormat, "bin") == 0)
	gsl_matrix_fwrite(dstStream, X);
      else
	gsl_matrix_fprintf(dstStream, X, "%f");
      fclose(dstStream);  

      // Free
      gsl_matrix_free(X);
    }
  delete mod;
  delete scheme;
  delete field;
  gsl_rng_free(r);
  freeConfig();

  return 0;
}
