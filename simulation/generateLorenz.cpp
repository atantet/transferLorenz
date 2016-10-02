#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "ODESolvers.hpp"

int main(int argc, char * argv[])
{
  // Output
  char dstExtension[] = "bin";
  //char dstExtension[] = "txt";

  // Model parameters
  //  char model[] = "randomLorenz";
  char model[] = "lorenz";
  double rho = 28.;
  double sigma = 10.;
  double beta = 8./3;
  size_t dim = 3;

  // Initial state
  gsl_vector * X0 = gsl_vector_alloc(dim);
  gsl_vector_set(X0, 0, 0.1);
  gsl_vector_set(X0, 1, 0.);
  gsl_vector_set(X0, 2, 0.);

  // Simulation parameters
  double dt = 1.e-3;
  int sampling = 1;
  double L = 1.e4;
  double spinup = 10;

  char postfix[128];
  char dstFileName[128];
  FILE *dstFile;
  sprintf(postfix, "_%s_rho%d_dt%d_sp%d_L%d", model, (int) (rho*1000),
	  (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)), sampling, (int) L);
  sprintf(dstFileName, "sim/sim%s.%s", postfix, dstExtension);
  dstFile = fopen(dstFileName, "w");

  // Result matrix
  gsl_matrix *X;

  // Numerical integration
  printf("Integrating simulation...\n");
  X = generateLorenzRK4(X0, rho, sigma, beta, L, dt, sampling, spinup);

  // Write results
  printf("Writing...\n");
  if (strcmp(dstExtension, "bin") == 0)
    gsl_matrix_fwrite(dstFile, X);
  else
    gsl_matrix_fprintf(dstFile, X, "%f");

  gsl_matrix_free(X);
  fclose(dstFile);  
		
  return 0;
}
