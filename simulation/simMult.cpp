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
#include <gsl_extension.hpp>
#include <omp.h>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simLorenz63.cpp 
 *  \ingroup examples
 *  \brief Simulate Lorenz (1963) deterministic flow.
 *
 *  Simulate Lorenz (1963) deterministic flow.
 */


bool isStationaryPoint(const gsl_matrix *X, const double gap,
		       const double printStep, const double tol);

  
/** \brief Simulation of Lorenz (1963) deterministic flow.
 *
 *  Simulation of Lorenz (1963) deterministic flow.
 *  After parsing the configuration file,
 *  the vector field of the Lorenz 1963 flow and the Runge-Kutta numerical scheme of order 4 are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  char dstPostfix[256], srcPostfix[256];

  // Read configuration file
  if (argc < 2) {
    std::cout << "Enter path to configuration file:" << std::endl;
    std::cin >> configFileName;
  }
  else
    strcpy(configFileName, argv[1]);
  try {
    Config cfg;
    std::cout << "Sparsing config file " << configFileName << std::endl;
    cfg.readFile(configFileName);
    readGeneral(&cfg);
    readModel(&cfg);
    readSimulation(&cfg);
    readSprinkle(&cfg);
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

  sprintf(srcPostfix, "_%s", caseName);
  sprintf(dstPostfix, "%s_rho%04d_L%d_spinup%d_dt%d_samp%d", srcPostfix,
	  (int) (p["rho"] * 100 + 0.1), (int) L, (int) spinup,
	  (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);
  
  // Iterate several trajectories
#pragma omp parallel
  {
    gsl_vector *init = gsl_vector_alloc(dim);
    gsl_matrix *X;
    char dstFileName[256];
    FILE *dstStream;
    size_t seed;

    // Define field
    vectorField *field = new Lorenz63(&p);
    // Define numerical scheme
    numericalScheme *scheme = new RungeKutta4(dim);
    // Define model (the initial state will be assigned later)
    model *mod = new model(field, scheme);

    // Set random number generator
    gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);
    // Get seed and set random number generator
    seed = (size_t) (1 + omp_get_thread_num());
#pragma omp critical
    {
      std::cout << "Setting random number generator with seed: " << seed
		<< std::endl;
    }
    gsl_rng_set(r, seed);

#pragma omp for
    for (size_t traj = 0; traj < (size_t) nTraj; traj++) {
      bool stat = true;
      while (stat) {
	// Get random initial distribution
	for (size_t d = 0; d < (size_t) dim; d++)
	  gsl_vector_set(init, d,
			 gsl_ran_flat(r, gsl_vector_get(minInitState, d),
				      gsl_vector_get(maxInitState, d)));

	// Set initial state
	printf("\nSetting initial state to (%.1lf, %.1lf, %.1lf)\n",
	       gsl_vector_get(init, 0),
	       gsl_vector_get(init, 1),
	       gsl_vector_get(init, 2));
	mod->setCurrentState(init);

	// Numerical integration of spinup
	std::cout << "Integrating spinup..." << std::endl;
	X = gsl_matrix_alloc(1, 1); // Fake allocation
	mod->integrate(init, spinup, dt, 0., printStepNum, &X);

	// Check if stationary point
	if (isStationaryPoint(X, 0.1, printStep, 1.e-8))
	  std::cout << "Trajectory converged to stationary point. Continue..."
		    << std::endl;
	else
	  stat = false;
      }

      // Numerical integration
      std::cout << "Integrating trajectory..." << std::endl;
      mod->integrate(L, dt, 0., printStepNum, &X);

      // Write results
#pragma omp critical
      {
	sprintf(dstFileName, "%s/simulation/sim%s_traj%d.%s",
		resDir, dstPostfix, (int) traj, fileFormat);
	if (!(dstStream = fopen(dstFileName, "w"))) {
	  std::cerr << "Can't open " << dstFileName
		    << " for writing simulation: " << std::endl;;
	  perror("");
	}

	std::cout << "Writing..." << std::endl;
	if (strcmp(fileFormat, "bin") == 0)
	  gsl_matrix_fwrite(dstStream, X);
	else
	  gsl_matrix_fprintf(dstStream, X, "%f");
	fclose(dstStream);  
      }

      // Free
      gsl_matrix_free(X);
    }
    delete mod;
    delete scheme;
    delete field;
    gsl_vector_free(init);
    gsl_rng_free(r);
  }
  
  freeConfig();

  return 0;
}


bool
isStationaryPoint(const gsl_matrix *X, const double gap,
		  const double printStep, const double tol) {
  const size_t nGap = (size_t) (gap / printStep + 0.1);
  double dist;
  gsl_vector *vec = gsl_vector_alloc(X->size2);
  gsl_vector_const_view vView
    = gsl_matrix_const_row(X, X->size1 - nGap - 1);

  gsl_matrix_get_row(vec, X, X->size1 - 1);
  gsl_vector_sub(vec, &vView.vector);
  dist = gsl_vector_get_norm(vec);
  gsl_vector_free(vec);

  return (dist < tol);
}
