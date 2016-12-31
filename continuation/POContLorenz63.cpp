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
#include <gsl_extension.hpp>
#include <math.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file POContLorenz63.cpp 
 *  \ingroup examples
 *  \brief Periodic orbit continuation in Lorenz 63 model.
 *
 *  Periodic orbit continuation in Lorenz 63 model.
 */


/* *  \brief Periodic orbit continuation in Lorenz 63 model.
 *
 *  Periodic orbit continuation in Lorenz 63 model.
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
     readContinuation(&cfg);
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

  gsl_matrix *solFM = gsl_matrix_alloc(dim + 1, dim + 1);
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_vector_complex *FloquetExp = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *FloquetVec = gsl_matrix_complex_alloc(dim, dim);
  gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(dim);
  gsl_matrix_view fm;
  char dstFileName[256], dstFileNameVec[256], dstFileNameExp[256],
    srcPostfix[256], dstPostfix[256];
  FILE *dstStream, *dstStreamVec, *dstStreamExp;


  // Define names and open destination file
  double contAbs = sqrt(contStep*contStep);
  double sign = contStep / contAbs;
  double exp = gsl_sf_log(contAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(contAbs) / exp);
  sprintf(srcPostfix, "_%s", caseName);
  sprintf(dstPostfix, "%s_cont%04d_contStep%de%d_dt%d_numShoot%d", srcPostfix,
	  (int) (gsl_vector_get(initCont, dim) * 1000 + 0.1),
	  (int) (mantis*1.01), (int) (exp*1.01),
	  (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), numShoot);
  sprintf(dstFileName, "%s/continuation/poCont%s.%s",
	  resDir, dstPostfix, fileFormat);

  if (!(dstStream = fopen(dstFileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing solution: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  sprintf(dstFileNameVec, "%s/continuation/poVecCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamVec = fopen(dstFileNameVec, "w")))
    {
      fprintf(stderr, "Can't open %s for writing Floquet vectors: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  sprintf(dstFileNameExp, "%s/continuation/poExpCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamExp = fopen(dstFileNameExp, "w")))
    {
      fprintf(stderr, "Can't open %s for writing Floquet exponents: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new Lorenz63Cont(sigma, beta);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianLorenz63Cont(sigma, beta, initCont);

  // Define numerical scheme
  //std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim + 1);
  //numericalScheme *scheme = new Euler(dim + 1);

  // Define model (the initial state will be assigned later)
  //std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Define linearized model 
  //std::cout << "Defining linearized model..." << std::endl;
  fundamentalMatrixModel *linMod = new fundamentalMatrixModel(mod, Jacobian);

  // Define periodic orbit problem
  periodicOrbitCont *track = new periodicOrbitCont(linMod, epsDist, epsStepCorrSize,
						   maxIter, dt, numShoot, verbose);

  // First correct
  std::cout << "Applying initial correction..." << std::endl;
  track->correct(initCont);

  if (!track->hasConverged())
    {
      std::cerr << "First correction could not converge." << std::endl;
      //return -1;
    }
  else
    {
      std::cout << "Found initial periodic orbit after "
		<< track->getNumIter() << " iterations"
		<< " with distance = " << track->getDist()
		<< " and step = " << track->getStepCorrSize() << std::endl;
      std::cout << "periodic orbit:" << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
      // Get solution and the fundamental matrix
      track->getCurrentState(initCont);
      track->getStabilityMatrix(solFM);
      fm = gsl_matrix_submatrix(solFM, 0, 0, dim, dim);

      // Find eigenvalues
      gsl_eigen_nonsymmv(&fm.matrix, eigVal, FloquetVec, w);
      // Convert to Floquet exponents
      gsl_vector_complex_log(FloquetExp, eigVal);
      gsl_vector_complex_scale_real(FloquetExp, 1. \
				    / gsl_vector_get(initCont, dim + 1));

      // Print periodic orbit
      std::cout << "periodic orbit:" << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
      std::cout << "Eigenvalues:" << std::endl;
      gsl_vector_complex_fprintf(stdout, FloquetExp, "%lf");
    }

  while ((gsl_vector_get(initCont, dim) >= contMin)
	 && (gsl_vector_get(initCont, dim) <= contMax))
    {
      // Find periodic orbit
      std::cout << "\nApplying continuation step..." << std::endl;
      track->continueStep(contStep);

      if (!track->hasConverged())
	{
	  std::cerr << "Continuation could not converge." << std::endl;
	  break;
	}
      else
	std::cout << "Found initial periodic orbit after "
		  << track->getNumIter() << " iterations"
		  << " with distance = " << track->getDist()
		  << " and step = " << track->getStepCorrSize() << std::endl;

      // Get solution and the fundamental matrix
      track->getCurrentState(initCont);
      track->getStabilityMatrix(solFM);
      fm = gsl_matrix_submatrix(solFM, 0, 0, dim, dim);

      // Find eigenvalues
      gsl_eigen_nonsymmv(&fm.matrix, eigVal, FloquetVec, w);
      // Convert to Floquet exponents
      gsl_vector_complex_log(FloquetExp, eigVal);
      gsl_vector_complex_scale_real(FloquetExp, 1. \
				    / gsl_vector_get(initCont, dim + 1));

      // Print periodic orbit
      std::cout << "periodic orbit:" << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
      std::cout << "Eigenvalues:" << std::endl;
      gsl_vector_complex_fprintf(stdout, FloquetExp, "%lf");

      // Write results
      if (strcmp(fileFormat, "bin") == 0)
	{
	  gsl_vector_fwrite(dstStream, initCont);
	  gsl_vector_complex_fwrite(dstStreamExp, FloquetExp);
	  gsl_matrix_complex_fwrite(dstStreamVec, FloquetVec);
	}
      else
	{
	  gsl_vector_fprintf(dstStream, initCont, "%lf");
	  gsl_vector_complex_fprintf(dstStreamExp, FloquetExp, "%lf");
	  gsl_matrix_complex_fprintf(dstStreamVec, FloquetVec, "%lf");
	}

      // Flush in case premature exit
      fflush(dstStream);
      fflush(dstStreamExp);
      fflush(dstStreamVec);
    }

  gsl_eigen_nonsymmv_free(w);
  gsl_vector_complex_free(eigVal);
  gsl_vector_complex_free(FloquetExp);
  gsl_matrix_complex_free(FloquetVec);
  delete track;
  delete linMod;
  delete mod;
  delete scheme;
  delete Jacobian;
  delete field;
  gsl_matrix_free(solFM);
  fclose(dstStreamExp);
  fclose(dstStreamVec);
  fclose(dstStream);  
  freeConfig();

  return 0;
}
