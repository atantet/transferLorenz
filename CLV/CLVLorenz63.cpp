#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cblas.h>
#include <gsl_extension.hpp>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file CLVLorenz63.cpp
 *  \ingroup examples
 *  \brief Compute CLVs in the Lorenz 63 model.
 *
 *  Compute the Covariant Lyapunov Vectors in Lorenz 63 model.
 */


/** \brief Compute CLVs in the Lorenz 63 model.
 *
 *  Compute the Covariant Lyapunov Vectors in Lorenz 63 model.
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

  char dstFileName[256], dstPostfix[256];
  FILE *dstStream;

  // Time related variables
  size_t nt = (size_t) (LCut / dt + 0.1);
  size_t ntSamp = nt / printStepNum;
  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63(rho, sigma, beta);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianLorenz63(rho, sigma, beta);

  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim);
  //numericalScheme *scheme = new Euler(dim);

  // Define model (the initial state will be assigned later)
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Define linearized model and initialize at initState (setting matrix to identity)
  std::cout << "Defining linearized model..." << std::endl;
  fundamentalMatrixModel *linMod = new fundamentalMatrixModel(mod, Jacobian, initState);


  // Allocation
  std::cout << "Allocating memory..." << std::endl;
  
  // Triangular matrix storage
  std::vector<gsl_matrix *> tri(ntSamp);
  // Backward and then Covariant Lyapunov vectors storage
  std::vector<gsl_matrix *> CLV(ntSamp);
  // Garbage matrix
  gsl_matrix *garbage = gsl_matrix_alloc(dim, dim);

  // Local stretching rates
  gsl_matrix *stretchRate = gsl_matrix_calloc(ntSamp, dim);
  // Lyapunov exponents
  gsl_vector *LyapExp = gsl_vector_calloc(dim);
  double Lyap;

  // Householder vector for Q
  gsl_vector *tau = gsl_vector_alloc(dim);
  // Orthogonal matrix
  gsl_matrix *Q = gsl_matrix_alloc(dim, dim);

  // Set random number generator
  gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);
  gsl_rng_set(r, gsl_vector_uint_get(seedRng, 0));

  // Propagate for the spinup period and update Jacobian and fundamental matrix
  std::cout << "Integrating for the spinup period..." << std::endl;
  linMod->mod->integrateForward(spinup, dt);
  linMod->setCurrentState();
  std::cout << "Updating model to current state:" << std::endl;
  gsl_vector_fprintf(stdout, linMod->mod->current, "%lf");


  // Go Forward
  std::cout << "Forward integration..." << std::endl;
  
  // Allocate and initialize random set of vectors
  tri[0] = gsl_matrix_alloc(dim, dim);
  CLV[0] = gsl_matrix_alloc(dim, dim);
  for (size_t i = 0; i < (size_t) dim; i++)
    for (size_t j = 0; j < (size_t) dim; j++)
      gsl_matrix_set(Q, i, j, gsl_ran_flat(r, -1., 1.));

  // Perform QR on current state matrix
  // to get initial set of orthogonal Backward Lyapunov Vector BLV(0)
  gsl_linalg_QR_decomp(Q, tau);
  // Unpack orthogonal and triangular matrices to
  // Backward Lyapunov Vectors and T
  // (tri is garbage and will be overwritten).
  gsl_linalg_QR_unpack(Q, tau, CLV[0], tri[0]);
  gsl_matrix_memcpy(Q, CLV[0]);

  for (size_t ktSamp = 1; ktSamp < ntSamp; ktSamp++)
    {
      // Allocate matrices
      tri[ktSamp] = gsl_matrix_alloc(dim, dim);
      CLV[ktSamp] = gsl_matrix_alloc(dim, dim);

      // Get time printStep propagator M(t, t-1) at current state
      linMod->integrateForward(printStep, dt);

      // Get BLVs BLV(t) from BLV(t-1)
      // by solving M(t, t-1) BLV(t-1) = BLV(t) T(t, t - 1)
      // (see (31) in Kuptsov & Parlitz, 2012).
      // Get Q(t) = M(t, t-1) BLV(t-1)
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., linMod->current,
		     CLV[ktSamp - 1], 0., Q);
      // QR decomposition of Q(t)
      gsl_linalg_QR_decomp(Q, tau);
      // Unpack to BLV(t) and T(t, t-1)
      gsl_linalg_QR_unpack(Q, tau, CLV[ktSamp], tri[ktSamp - 1]);
      gsl_matrix_memcpy(Q, CLV[ktSamp]);
      

      // Compute the Lyapunov exponents
      for (size_t i = 0; i < (size_t) dim; i++)
	{
	  Lyap = gsl_sf_log(sqrt(gsl_pow_2(gsl_matrix_get(tri[ktSamp - 1], i, i))));
	  
	  // Integrate the Lyapunov exponents
	  gsl_vector_set(LyapExp, i, gsl_vector_get(LyapExp, i) + Lyap);
	}
	  
      // Update Jacobian to current model state (already iterated)
      // and set fundamental matrix to identity
      linMod->setCurrentState();
    }
  
  // Get Last trigangular matrix (already allocated)
  // Get M(tf + 1, tf)
  linMod->integrateForward(printStep, dt);
  // Get Q(tf + 1) = M(tf + 1, tf) BLV(tf)
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., linMod->current,
		 CLV[ntSamp - 1], 0., Q);
  // Get BLV(tf + 1) T(tf + 1, tf) = Q(tf + 1) = M(tf + 1, tf) BLV(tf)
  gsl_linalg_QR_decomp(Q, tau);
  gsl_linalg_QR_unpack(Q, tau, garbage, tri[ntSamp-1]);
  // Compute the Lyapunov exponents
  for (size_t i = 0; i < (size_t) dim; i++)
    {
      Lyap = gsl_sf_log(sqrt(gsl_pow_2(gsl_matrix_get(tri[ntSamp - 1], i, i))));
      
      // Integrate the Lyapunov exponents
      gsl_vector_set(LyapExp, i, gsl_vector_get(LyapExp, i) + Lyap);
    }
  // Divide Lyapunov exponents by length
  gsl_vector_scale(LyapExp, 1. / LCut);
	  

  
  // Print
  std::cout << "Final state:" << std::endl;
  gsl_vector_fprintf(stdout, linMod->mod->current, "%lf");
  std::cout << "Lyapunov exponents: " << std::endl;
  gsl_vector_fprintf(stdout, LyapExp, "%lf");

  
  // Get CLVs
  std::cout << "Backward integration..." << std::endl;
  // Initialize random upper triangular matrix Q(tf + 1) = Am(tf + 1)
  gsl_matrix_set_zero(Q);
  double coljNorm, elem;
  for (size_t j = 0; j < (size_t) dim; j++)
    {
      coljNorm = 0.;
      // Set upper triangle coefficients randomly
      for (size_t i = 0; i <= j; i++)
	{
	  elem = gsl_ran_flat(r, -1., 1.);
	  gsl_matrix_set(Q, i, j, elem);
	  coljNorm += elem*elem;
	}
      coljNorm = sqrt(coljNorm);
      
      // Normalize column j
      for (size_t i = 0; i <= j; i++)
	gsl_matrix_set(Q, i, j, gsl_matrix_get(Q, i, j) / coljNorm);
    }

  // Step backward to unfold CLVs
  gsl_vector_view view;
  double stretch;
  for (int ktSamp = (int) (ntSamp-1); ktSamp >= 0; ktSamp--)
    {
      // Get Am(t) = tri(t, t+1)^{-1} Am(t+1)
      // (see (50) in Kuptsov & Parlitz, 2012, with C(t, t+1) = I).
      gsl_blas_dtrsm(CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 1.,
		     tri[ktSamp], Q);

      for (size_t j = 0; j < (size_t) dim; j++)
	{
	  // Get norm of column j of Q
	  view = gsl_matrix_column(Q, j);
	  coljNorm = gsl_vector_get_norm(&view.vector);

	  // Normalize column j of Q
	  gsl_vector_scale(&view.vector, 1. / coljNorm);

	  // Get local stretching rate
	  stretch = -gsl_sf_log(coljNorm) / printStep;
	  gsl_matrix_set(stretchRate, ktSamp, j,
			 gsl_matrix_get(stretchRate, ktSamp, j) + stretch);
	}

      // Get CLV[t] = BLV(t) * Am(t)
      // (see (43) in Kuptsov & Parlitz, 2012).
      // Using BLAS matrix triangular matrix product.
      // In place -> overwrites backward by covariant Lyapunov vectors !!!
      gsl_blas_dtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1.,
		     Q, CLV[ktSamp]);
    }

  // Define names and open destination file
  std::cout << "Writing results..." << std::endl;
  sprintf(dstPostfix, "%s_L%d_spinup%d_dt%d_samp%d", srcPostfixModel, (int) L,
	  (int) spinup, (int) (round(-gsl_sf_log(dt)/gsl_sf_log(10)) + 0.1),
	  (int) printStepNum);
  
  // Save Lyapunov exponents
  sprintf(dstFileName, "%s/CLV/LyapExp%s.%s", resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      std::cerr << "Can't open " << dstFileName
		<< " for writing Lyapunov exponents: " << std::endl;;
      perror("");
      return EXIT_FAILURE;
    }
  if (strcmp(fileFormat, "bin") == 0)
    gsl_vector_fwrite(dstStream, LyapExp);
  else
    gsl_vector_fprintf(dstStream, LyapExp, "%lf");  
  fclose(dstStream);

  // Save Stretching rates
  sprintf(dstFileName, "%s/CLV/stretchRate%s.%s", resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      std::cerr << "Can't open " << dstFileName
		<< " for writing stretching rates: " << std::endl;;
      perror("");
      return EXIT_FAILURE;
    }
  if (strcmp(fileFormat, "bin") == 0)
    gsl_matrix_fwrite(dstStream, stretchRate);
  else
    gsl_matrix_fprintf(dstStream, stretchRate, "%lf");
  fclose(dstStream);
  
  // Save covariant Lyapunov vectors
  sprintf(dstFileName, "%s/CLV/CLV%s.%s", resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      std::cerr << "Can't open " << dstFileName
		<< " for writing covariant Lyapunov vectors: " << std::endl;;
      perror("");
      return EXIT_FAILURE;
    }
  for (size_t ktSamp = 0; ktSamp < ntSamp; ktSamp++)
    if (strcmp(fileFormat, "bin") == 0)
      gsl_matrix_fwrite(dstStream, CLV[ktSamp]);
    else
      gsl_matrix_fprintf(dstStream, CLV[ktSamp], "%lf");
  fclose(dstStream);

  // Free
  gsl_rng_free(r);
  gsl_matrix_free(garbage);
  gsl_matrix_free(Q);
  for (size_t ktSamp = 0; ktSamp < ntSamp; ktSamp++)
    {
      gsl_matrix_free(tri[ktSamp]);
      gsl_matrix_free(CLV[ktSamp]);
    }
  gsl_vector_free(LyapExp);
  gsl_matrix_free(stretchRate);
  gsl_vector_free(tau);
  delete linMod;
  delete mod;
  delete scheme;
  delete Jacobian;
  delete field;
  freeConfig();

  return 0;
}
