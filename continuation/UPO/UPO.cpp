#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <list>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl_extension.hpp>
#include <ODESolvers.hpp>
#include "../cfg/readConfig.hpp"


/** \file UPO.cpp
 *  \brief Get Unstable Periodic Oribits (UPOs) from a model and a time series.
 * 
 * Get Unstable Periodic Oribits (UPOs) from a model and a time series.
 */


// Definitions 
/** \brief Check if the trajectory has crossed the Poincare section. */
bool poincareZPlane(gsl_matrix *traj, size_t idx);

// Definitions 
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
  size_t seed = gsl_vector_uint_get(seedRng, 0);
  char *postfix[256];

  

  // Define model
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63(rho, sigma, beta);
  
  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim, dt);

  // Define model (the initial state will be assigned later)
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  
  // Allocate trajectory and grid limits
  traj = gsl_matrix_alloc(nt0, dim);

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

  // Read time series
  std::cout << "Reading trajectory in " << srcFileName << std::endl;
  if (strcmp(fileFormat, "bin") == 0)
    gsl_matrix_fread(srcStream, traj);
  else
    gsl_matrix_fscanf(srcStream, traj);


  // Mark points on the Poincare section
  std::list<size_t> idxSection; 
  std::list<size_t>::iterator it;
  for (size_t t = 0; t < nt0; t++)
    if (poincareZPlane(traj, t))
      idxSection.push_back(t);

  // Initialize initial and final states and fundamental matrices
  // of an orbit
  gsl_vector *x0 = gsl_vector_alloc(dim);
  gsl_matrix *M0 = gsl_matrix_alloc(dim, dim);
  gsl_vector *xt = gsl_vector_alloc(dim);
  gsl_matrix *Mt = gsl_matrix_alloc(dim, dim);
  double T0;
  
  // Find first period 1 orbits
  it = idxSection.begin();
  T0 = -(*it);
  gsl_matrix_get_row(x0, traj, *it);
  it++;
  T0 += (*it);
  gsl_matrix_get_row(xt, traj, *it);
  T0 *= dt;
  // Set initial fundamental matrix to identity
  gsl_matrix_set_identity(M0);

  // Get UPOp
  double errDist = 1.e27;
  double errDelta = 1.e27;
  double eps = 1.e-4;
  size_t maxIter = 1000;
  size_t nIter = 0;
  while (((errDist > eps) || (errDelta > eps)) && (nIter < maxIter))
    {
      std::cout << "\n---Iteration " << nIter << "---\n"
		<<  "T0 = " << T0
		<< "x0 = " << std::endl;
      gsl_vector_fprintf(stdout, x0, "%.3f");
	//	   <<  'x0 = ', x0[0]
    }
  
  // // Perform Newton step
  // (step, errDist, errDelta, xt, Mt)				
  //   = NewtonStepMulti(x0, M0, T0, field, Jacobian, p, dt0);


  // Free and close
  gsl_matrix_free(Mt);
  gsl_vector_free(xt);
  gsl_matrix_free(M0);
  gsl_vector_free(x0);
  gsl_matrix_free(traj);
  fclose(srcStream);
		
  return 0;
}

/**
 * Function returning true if the trajectory has crossed the Poincare section.
 * \param[in] traj Trajectory.
 * \param[in] idx  Index of the trajectory at which to check for crossing.
 * \return         True if the trajectory is crossing the Poincare section.
 */
bool poincareZPlane(gsl_matrix *traj, size_t idx)
{
  if ((gsl_matrix_get(traj, idx+1, 2) >= rho - 1)
      && (gsl_matrix_get(traj, idx, 2) < rho - 1))
    return true;
  else
    return false;
}
