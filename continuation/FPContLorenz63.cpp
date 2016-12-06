#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <math.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file POContLorenz63.cpp 
 *  \ingroup examples
 *  \brief Fixed point continuation in Lorenz 63 model.
 *
 *  Fixed point continuation in Lorenz 63 model.
 */


/* *  \brief Fixed point continuation in Lorenz 63 model.
 *
 *  Fixed point continuation in Lorenz 63 model.
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

  gsl_matrix *solJac = gsl_matrix_alloc(dim + 1, dim + 1);
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_eigen_nonsymm_workspace *w = gsl_eigen_nonsymm_alloc(dim);
  gsl_matrix_view jac;
  char dstFileName[256], dstFileNameJac[256], dstFileNameEig[256], dstPostfix[256];
  FILE *dstStream, *dstStreamJac, *dstStreamEig;


  // Define names and open destination file
  double contAbs = sqrt(contStep*contStep);
  double sign = contStep / contAbs;
  double exp = gsl_sf_log(contAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(contAbs) / exp);
  sprintf(dstPostfix, "%s_cont%04d_contStep%de%d", srcPostfix,
	  (int) (gsl_vector_get(initCont, dim) * 1000 + 0.1),
	  (int) (mantis*1.01), (int) (exp*1.01));
  sprintf(dstFileName, "%s/continuation/fpCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  sprintf(dstFileNameJac, "%s/continuation/fpJacCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamJac = fopen(dstFileNameJac, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  sprintf(dstFileNameEig, "%s/continuation/fpEigCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamEig = fopen(dstFileNameEig, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63Cont(sigma, beta);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianLorenz63Cont(sigma, beta, initCont);

  // Define fixed point problem
  fixedPointCont *track = new fixedPointCont(field, Jacobian, eps, eps, maxIter);

  // First correct
  std::cout << "Applying initial correction..." << std::endl;
  track->correct(initCont);

  if (!track->hasConverged())
    {
      std::cerr << "First correction could not converge." << std::endl;
      return -1;
    }
  else
    std::cout << "Found initial fixed point after "
	      << track->getNumIter() << " iterations"
	      << " with distance = " << track->getDist()
	      << " and step = " << track->getStepCorrSize() << std::endl;


  while ((gsl_vector_get(initCont, dim) >= contMin)
	 && (gsl_vector_get(initCont, dim) <= contMax))
    {
      // Find fixed point
      std::cout << "\nApplying continuation step..." << std::endl;
      track->continueStep(contStep);

      if (!track->hasConverged())
	{
	  std::cerr << "Continuation could not converge." << std::endl;
	  break;
	}
      else
	std::cout << "Found initial fixed point after "
		  << track->getNumIter() << " iterations"
		  << " with distance = " << track->getDist()
		  << " and step = " << track->getStepCorrSize() << std::endl;

      // Get solution and the Jacobian
      track->getCurrentState(initCont);
      track->getStabilityMatrix(solJac);
      jac = gsl_matrix_submatrix(solJac, 0, 0, dim, dim);

      // Find eigenvalues
      gsl_eigen_nonsymm(&jac.matrix, eigVal, w);

      // Print fixed point
      std::cout << "Fixed point:" << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
      std::cout << "Eigenvalues:" << std::endl;
      gsl_vector_complex_fprintf(stdout, eigVal, "%lf");

      // Write results
      gsl_vector_fprintf(dstStream, initCont, "%lf");
      gsl_vector_complex_fprintf(dstStreamEig, eigVal, "%lf");
      gsl_matrix_fprintf(dstStreamJac, &jac.matrix, "%lf");
    }
  
  gsl_eigen_nonsymm_free(w);
  gsl_vector_complex_free(eigVal);
  delete track;
  delete Jacobian;
  delete field;
  gsl_matrix_free(solJac);
  gsl_vector_free(initCont);
  fclose(dstStreamEig);
  fclose(dstStreamJac);
  fclose(dstStream);  
  freeConfig();

  return 0;
}
