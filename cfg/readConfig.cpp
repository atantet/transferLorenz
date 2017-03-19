#include "../cfg/readConfig.hpp"

// Configuration variables
char resDir[256];               //!< Root directory in which results are written
char caseName[256];             //!< Name of the case
char caseNameModel[256];        //!< Name of the case to simulate 
double rho;                     //!< Parameters for the Lorenz flow
double sigma;                   //!< Parameters for the Lorenz flow
double beta;                    //!< Parameters for the Lorenz flow
char fileFormat[256];           //!< File format of output ("txt" or "bin")
int dim;                        //!< Dimension of the phase space
// Continuation
double epsDist;                 //!< Tracking distance tolerance
double epsStepCorrSize;         //!< Tracking correction step size tolerance
int maxIter;                    //!< Maximum number of iterations for correction
int numShoot;                   //!< Number of shoots
double contStep;                //!< Step size of parameter for continuation
double contMin;                 //!< Lower limit to which to continue
double contMax;                 //!< Upper limit to which to continue
bool verbose;                   //!< Verbose mode selection
gsl_vector *initCont;           //!< Initial state for continuation
// Simulation
gsl_vector *initState;          //!< Initial state for simulation
double LCut;                    //!< Length of the time series without spinup
double spinup;                  //!< Length of initial spinup period to remove
double L;                       //!< Total length of integration
double dt;                      //!< Time step of integration
double printStep;               //!< Time step of output
size_t printStepNum;            //!< Time step of output in number of time steps of integration
// Sprinkle
int nTraj;                      //!< Number of trajectories to sprinkle
gsl_vector *minInitState;       //!< Lower limits of initial states of trajectories
gsl_vector *maxInitState;       //!< Upper limits of initial states of trajectories
gsl_vector_uint *seedRng;       //!< Seeds used to initialize the simulations
size_t nSeeds;                  //!< Number of seeds
char boxPostfix[256];           //!< Postfix associated with the bounding box
// Embedding
size_t nt0;                     //!< Number of time steps of the source time series
size_t nt;                      //!< Number of time steps of the observable
int dimObs;                     //!< Dimension of the observable
size_t embedMax;                //!< Maximum lag for the embedding
gsl_vector_uint *components;    //!< Components in the time series used by the observable
gsl_vector_uint *embedding;     //!< Embedding lags for each component
// Grid
char gridPostfix[256];          //!< Postfix associated with the grid
bool readGridMem;               //!< Whether to read the grid membership vector
size_t N;                       //!< Dimension of the grid
gsl_vector_uint *nx;            //!< Number of grid boxes per dimension
gsl_vector *gridLimitsLow;      //!< Grid limits
gsl_vector *gridLimitsUp;       //!< Grid limits
char gridLimitsType[32];        //!< Grid limits type
size_t nLags;                   //!< Number of transition lags for which to calculate the spectrum
gsl_vector *tauRng;             //!< Lags for which to calculate the spectrum
int nev;                        //!< Number of eigenvectors to calculate
char obsName[256];              //!< Name associated with the observable
configAR config;                //!< Configuration data for the eigen problem
/** Declare default structure looking for largest magnitude eigenvalues */
char configFileName[256];       //!< Name of the configuration file
bool stationary;                //!< Whether the problem is stationary or not
bool getForwardEigenvectors;    //!< Whether to get forward eigenvectors
bool getBackwardEigenvectors;   //!< Whether to get backward eigenvectors
bool makeBiorthonormal;         //!< Whether to make eigenvectors biorthonormal

/** \file readConfig.cpp
 *  \brief Definitions for readConfig.hpp
 */


/**
 * Read general configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readGeneral(const Config *cfg, const bool verboseCFG)
{
  if (verboseCFG)
    std::cout << std::endl << "---general---" << std::endl;
  
  strcpy(resDir, (const char *) cfg->lookup("general.resDir"));
  // Output format
  strcpy(fileFormat, (const char *) cfg->lookup("general.fileFormat"));
  if (verboseCFG) {
    std::cout << "Results directory: " << resDir << std::endl;
    std::cout << "Output file format: " << fileFormat << std::endl;
  }
    
  return;
}

/**
 * Read model configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readModel(const Config *cfg, const bool verboseCFG)
{
  if (cfg->exists("model"))
    {
      /** Get model settings */
      if (verboseCFG)
	std::cout << std::endl << "---model---" << std::endl;

      // Dimension
      dim = cfg->lookup("model.dim");
      dimObs = dim;
      if (verboseCFG)
	std::cout << "dim = " << dim << std::endl;

      // Case name
      strcpy(caseName, (const char *) cfg->lookup("model.caseName"));
      if (verboseCFG)
	std::cout << "Case name: " << caseName << std::endl;
      if (cfg->exists("model.rho"))
	{
	  rho = cfg->lookup("model.rho");
	  if (verboseCFG)
	    std::cout << "rho = " << rho << std::endl;
	}
      if (cfg->exists("model.sigma"))
	{
	  sigma = cfg->lookup("model.sigma");
	  if (verboseCFG)
	    std::cout << "sigma = " << sigma << std::endl;
	}
      if (cfg->exists("model.beta"))
	{
	  beta = cfg->lookup("model.beta");
	  if (verboseCFG)
	    std::cout << "beta = " << beta << std::endl;
	}
      sprintf(caseNameModel, "%s_rho%d_sigma%d_beta%d", caseName,
	      (int) (rho * 1000 + 0.1), (int) (sigma * 1000 + 0.1),
	      (int) (beta * 1000 + 0.1));
    }
  else
    if (verboseCFG)
      std::cout << "Model configuration section does not exist..." << std::endl;

  return;
}


/**
 * Read continuation configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readContinuation(const Config *cfg, const bool verboseCFG)
{
  /** Get continuation settings */
  if (cfg->exists("continuation"))
    {
      epsDist = cfg->lookup("continuation.epsDist");
      epsStepCorrSize = cfg->lookup("continuation.epsStepCorrSize");
      maxIter = cfg->lookup("continuation.maxIter");
      numShoot = cfg->lookup("continuation.numShoot");
      contStep = cfg->lookup("continuation.contStep");
      contMin = cfg->lookup("continuation.contMin");
      contMax = cfg->lookup("continuation.contMax");
      verbose = cfg->lookup("continuation.verbose");
      if (verboseCFG) {
	std::cout << "\n" << "---continuation---" << std::endl;
	std::cout << "epsDist = " << epsDist << std::endl;
	std::cout << "epsStepCorrSize = " << epsStepCorrSize << std::endl;
	std::cout << "maxIter = " << maxIter << std::endl;
	std::cout << "numShoot = " << numShoot << std::endl;
	std::cout << "contStep = " << contStep << std::endl;
	std::cout << "contMin = " << contMin << std::endl;
	std::cout << "contMax = " << contMax << std::endl;
	std::cout << "verbose = " << verbose << std::endl;
      }
      // Initial continuation state (dim+1 for fp, dim+2 for po)
      const Setting &initContSetting = cfg->lookup("continuation.initCont");
      initCont = gsl_vector_alloc(initContSetting.getLength());
      if (verboseCFG)
	std::cout << "initCont = [";
      for (size_t i =0; i < (size_t) (initContSetting.getLength()); i++)
	{
	  gsl_vector_set(initCont, i, initContSetting[i]);
	  if (verboseCFG)
	    std::cout << gsl_vector_get(initCont, i) << " ";
	}
      if (verboseCFG)
	std::cout << "]" << std::endl;
    }
  else
    if (verboseCFG)
      std::cout << "Continuation configuration section does not exist."
		<< std::endl;
    
  return;
}


/**
 * Read simulation configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readSimulation(const Config *cfg, const bool verboseCFG)
{
  if (cfg->exists("simulation"))
    {
      /** Get simulation settings */
      if (verboseCFG)
	std::cout << "\n" << "---simulation---" << std::endl;

      // Initial state
      if (cfg->exists("simulation.initState"))
	{
	  const Setting &initStateSetting =
	    cfg->lookup("simulation.initState");
	  initState = gsl_vector_alloc(dim);
	  if (verboseCFG)
	    std::cout << "initState = [";
	  for (size_t i =0; i < (size_t) (initStateSetting.getLength()); i++)
	    {
	      gsl_vector_set(initState, i, initStateSetting[i]);
	      if (verboseCFG)
		std::cout << gsl_vector_get(initState, i) << " ";
	    }
	  if (verboseCFG)
	    std::cout << "]" << std::endl;
	}

      // Simulation length without spinup
      LCut = cfg->lookup("simulation.LCut");

      // Time step
      dt = cfg->lookup("simulation.dt");

      // Spinup period to remove
      spinup = 0.;
      if (cfg->exists("simulation.spinup"))
	spinup = cfg->lookup("simulation.spinup");
	    
      // Sub-printStep
      printStep = dt;
      if (cfg->exists("simulation.printStep"))
	printStep = cfg->lookup("simulation.printStep");

      L = LCut + spinup;
      printStepNum = (size_t) (printStep / dt + 0.1);
      nt0 = (size_t) (LCut / printStep + 0.1 + 1.);
      // Add 1 for the initial state

      if (verboseCFG) {
	std::cout << "LCut = " << LCut << std::endl;
	std::cout << "dt = " << dt << std::endl;
	std::cout << "spinup = " << spinup << std::endl;
	std::cout << "printStep = " << printStep << std::endl;
      }
    }
  else
    if (verboseCFG)
      std::cout << "Simulation configuration section does not exist."
		<< std::endl;

  return;
}

/**
 * Read sprinkle configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readSprinkle(const Config *cfg, const bool verboseCFG)
{
  char cpyBuffer[256];
  
  // Get sprinkle settings
  if (cfg->exists("sprinkle"))
    {
      // Number of trajectories
      nTraj = cfg->lookup("sprinkle.nTraj");

      if (verboseCFG) {
	std::cout << std::endl << "---sprinkle---" << std::endl;
	std::cout << "nTraj = " << nTraj << std::endl;
      }

      // Min value of state
      const Setting &minInitStateSetting =
	cfg->lookup("sprinkle.minInitState");
      minInitState = gsl_vector_alloc(dim);
      if (verboseCFG)
	std::cout << "minInitState = {";
      for (size_t d = 0; d < (size_t) dim; d++) {
	gsl_vector_set(minInitState, d, minInitStateSetting[d]);
	if (verboseCFG)
	  std::cout << gsl_vector_get(minInitState, d) << ", ";
      }
      if (verboseCFG)
	std::cout << "}" << std::endl;

      // Max value of state
      const Setting &maxInitStateSetting =
	cfg->lookup("sprinkle.maxInitState");
      maxInitState = gsl_vector_alloc(dim);
      if (verboseCFG)
	std::cout << "maxInitState = {";
      for (size_t d = 0; d < (size_t) dim; d++) {
	gsl_vector_set(maxInitState, d, maxInitStateSetting[d]);
	if (verboseCFG)
	  std::cout << gsl_vector_get(maxInitState, d) << ", ";
      }
      if (verboseCFG)
	std::cout << "}" << std::endl;

      // Seeds
      const Setting &seedRngSetting = cfg->lookup("sprinkle.seedRng");
      nSeeds = seedRngSetting.getLength();
      seedRng = gsl_vector_uint_alloc(nSeeds);
      if (verboseCFG)
	std::cout << "seedRng = {";
      for (size_t seed = 0; seed < nSeeds; seed++) {
	gsl_vector_uint_set(seedRng, seed, seedRngSetting[seed]);
	if (verboseCFG)
	  std::cout << gsl_vector_uint_get(seedRng, seed) << ", ";
      }
      if (verboseCFG)
	std::cout << "}" << std::endl;

      // Define box postfix
      sprintf(boxPostfix, "");
      for (size_t d = 0; d < (size_t) dimObs; d++)
	{
	  if (minInitState && maxInitState)
	    {
	      strcpy(cpyBuffer, boxPostfix);
	      sprintf(boxPostfix, "%s_l%dh%d", cpyBuffer,
		      (int) gsl_vector_get(minInitState, d),
		      (int) gsl_vector_get(maxInitState, d));
	    }
	  else
	    {
	      strcpy(cpyBuffer, boxPostfix);
	      sprintf(boxPostfix, "%s_minmax", cpyBuffer);
	    }
	}
    }
  else
    if (verboseCFG)
      std::cout << "Sprinkle configuration section does not exist."
		<< std::endl;
    
   return;
}


/**
 * Read observable configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readObservable(const Config *cfg, const bool verboseCFG)
{
  char cpyBuffer[256];

  // Get observable settings
  if (cfg->exists("observable"))
    {
      if (verboseCFG)
	std::cout << std::endl << "---observable---" << std::endl;
    
      // Components
      const Setting &compSetting = cfg->lookup("observable.components");
      dimObs = compSetting.getLength();
      components = gsl_vector_uint_alloc(dimObs);
      if (verboseCFG)
	std::cout << "Components: [";
      for (size_t d = 0; d < (size_t) dimObs; d++)
	{
	  gsl_vector_uint_set(components, d, compSetting[d]);
	  if (verboseCFG)
	    std::cout << gsl_vector_uint_get(components, d) << " ";
	}
      if (verboseCFG)
	std::cout << "]" << std::endl;

      // Embedding
      const Setting &embedSetting = cfg->lookup("observable.embeddingDays");
      embedding = gsl_vector_uint_alloc(dimObs);
      if (verboseCFG)
	std::cout << "Embedding: [";
      for (size_t d = 0; d < (size_t) dimObs; d++)
	{
	  double embd = embedSetting[d];
	  gsl_vector_uint_set(embedding, d,
			      (int) nearbyint(embd / printStep));
	  if (verboseCFG)
	    std::cout << embd << " ";
	  sprintf(cpyBuffer, obsName);
	  sprintf(obsName, "%s_c%d_e%d", cpyBuffer,
		  (int) gsl_vector_uint_get(components, d), (int) embd);
	}
      if (verboseCFG)
	std::cout << "]" << std::endl;
      embedMax = gsl_vector_uint_max(embedding);
    }
  else
    {
      dimObs = dim;
      embedMax = 0;
      if (verboseCFG)
	std::cout << "Observable configuration section does not exist."
		  << std::endl;
    }
  nt = nt0 - embedMax;			    
    
  return;
}


/**
 * Read grid configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readGrid(const Config *cfg, const bool verboseCFG)
{
  char cpyBuffer[256];

  // Get grid settings
  if (cfg->exists("grid"))
    {
      if (verboseCFG)
	std::cout << std::endl << "---grid---" << std::endl;
      const Setting &nxSetting = cfg->lookup("grid.nx");
      nx = gsl_vector_uint_alloc(dimObs);
      N = 1;
      if (verboseCFG)
	std::cout << "Number of grid boxes per dimension:" << std::endl;
      for (size_t d = 0; d < (size_t) (dimObs); d++)
	{
	  gsl_vector_uint_set(nx, d, nxSetting[d]);
	  N *= gsl_vector_uint_get(nx, d);
	  if (verboseCFG)
	    std::cout << "dim " << d+1 << ": "
		      << gsl_vector_uint_get(nx, d) << std::endl;
	}
    
      // Grid limits type
      strcpy(gridLimitsType, (const char *)
	     cfg->lookup("grid.gridLimitsType"));
      if (verboseCFG)
	std::cout << "Grid limits type: " << gridLimitsType << std::endl;

      // Grid limits
      if (cfg->exists("grid.gridLimits"))
	{
	  const Setting &gridLimitsLowSetting
	    = cfg->lookup("grid.gridLimitsLow");
	  const Setting &gridLimitsUpSetting
	    = cfg->lookup("grid.gridLimitsUp");
	  gridLimitsLow = gsl_vector_alloc(dimObs);
	  gridLimitsUp = gsl_vector_alloc(dimObs);
	  if (verboseCFG)
	    std::cout << "Grid limits (low, high):" << std::endl;
	  for (size_t d = 0; d < (size_t) (dimObs); d++)
	    {
	      gsl_vector_set(gridLimitsLow, d, gridLimitsLowSetting[d]);
	      gsl_vector_set(gridLimitsUp, d, gridLimitsUpSetting[d]);
	      if (verboseCFG)
		std::cout << "dim " << d+1 << ": ("
			  << gsl_vector_get(gridLimitsLow, d) << ", "
			  << gsl_vector_get(gridLimitsUp, d) << ")"
			  << std::endl;
	    }
	}
      else
	{
	
	  gridLimitsLow = gsl_vector_alloc(dimObs);
	  gridLimitsUp = gsl_vector_alloc(dimObs);
	  gsl_vector_memcpy(gridLimitsLow, minInitState);
	  gsl_vector_memcpy(gridLimitsUp, maxInitState);
	  if (verboseCFG)
	    std::cout << "Grid limits (low, high):" << std::endl;
	  for (size_t d = 0; d < (size_t) (dimObs); d++)
	    {
	      if (verboseCFG)
		std::cout << "dim " << d+1 << ": ("
			  << gsl_vector_get(gridLimitsLow, d) << ", "
			  << gsl_vector_get(gridLimitsUp, d) << ")"
			  << std::endl;
	    }
	}
    
      if (cfg->exists("grid.readGridMem"))
	readGridMem = cfg->lookup("grid.readGridMem");
      else
	readGridMem = false;
      if (verboseCFG)
	std::cout << "readGridMem: " << readGridMem << std::endl;
      
      // Define grid name
      sprintf(gridPostfix, "");
      for (size_t d = 0; d < (size_t) dimObs; d++)
	{
	  if (gridLimitsLow && gridLimitsUp)
	    {
	      strcpy(cpyBuffer, gridPostfix);
	      sprintf(gridPostfix, "%s_n%dl%dh%d", cpyBuffer,
		      gsl_vector_uint_get(nx, d),
		      (int) gsl_vector_get(gridLimitsLow, d),
		      (int) gsl_vector_get(gridLimitsUp, d));
	    }
	  else
	    {
	      strcpy(cpyBuffer, gridPostfix);
	      sprintf(gridPostfix, "%s_n%dminmax", cpyBuffer,
		      gsl_vector_uint_get(nx, d));
	    }
	}
      strcpy(cpyBuffer, gridPostfix);    
    }
  else
    if (verboseCFG)
      std::cout << "Grid configuration section does not exist."
		<< std::endl;


  return;
}


/**
 * Read transfer configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readTransfer(const Config *cfg, const bool verboseCFG)
{
  // Get transition settings
  if (cfg->exists("transfer"))
    {
      const Setting &tauRngSetting = cfg->lookup("transfer.tauRng");
      nLags = tauRngSetting.getLength();
      tauRng = gsl_vector_alloc(nLags);

      if (verboseCFG) {
	std::cout << std::endl << "---transfer---" << std::endl;
	std::cout << "tauRng = [";
      }
      for (size_t lag = 0; lag < nLags; lag++) {
	gsl_vector_set(tauRng, lag, tauRngSetting[lag]);
	if (verboseCFG)
	  std::cout << gsl_vector_get(tauRng, lag) << " ";
      }
      if (verboseCFG)
	std::cout << "]" << std::endl;

      stationary = cfg->lookup("transfer.stationary");
      if (verboseCFG)
	std::cout << "Is stationary: " << stationary << std::endl;
    }
  else
      if (verboseCFG)
	std::cout << "Transfer configuration section does not exist."
		  << std::endl;

  return;
}


/**
 * Read spectrum configuration section.
 * \param[in] cfg Pointer to configuration object.
 */
void
readSpectrum(const Config *cfg, const bool verboseCFG)
{
  configAR defaultCfgAR = {"LM", 0, 0., 0, NULL, true};
  
  if (cfg->exists("spectrum"))
    {
      // Get spectrum setting 
      nev = cfg->lookup("spectrum.nev");
      if (verboseCFG)
	std::cout << std::endl << "---spectrum---" << std::endl;
      // Get eigen problem configuration
      config = defaultCfgAR;
      if (cfg->exists("spectrum.which"))
	{
	  strcpy(config.which, (const char *) cfg->lookup("spectrum.which"));
	}
      if (cfg->exists("spectrum.ncv"))
	{
	  config.ncv = cfg->lookup("spectrum.ncv");
	}
      if (cfg->exists("spectrum.tol"))
	{
	  config.tol = cfg->lookup("spectrum.tol");
	}
      if (cfg->exists("spectrum.maxit"))
	{
	  config.maxit = cfg->lookup("spectrum.maxit");
	}
      if (cfg->exists("spectrum.AutoShift"))
	{
	  config.AutoShift = (bool) cfg->lookup("spectrum.AutoShift");
	}
      if (verboseCFG) {
	std::cout << "nev: " << nev << std::endl;
	std::cout << "which: " << config.which << std::endl;
	std::cout << "ncv: " << config.ncv << std::endl;
	std::cout << "tol: " << config.tol << std::endl;
	std::cout << "maxit: " << config.maxit << std::endl;
	std::cout << "AutoShift: " << config.AutoShift << std::endl;
	std::cout << std::endl;
      }

      if (cfg->exists("spectrum.getForwardEigenvectors"))
	{
	  getForwardEigenvectors =
	    cfg->lookup("spectrum.getForwardEigenvectors");
	}
      if (cfg->exists("spectrum.getBackwardEigenvectors"))
	{
	  getBackwardEigenvectors =
	    cfg->lookup("spectrum.getBackwardEigenvectors");
	}
      if (cfg->exists("spectrum.makeBiorthonormal"))
	{
	  makeBiorthonormal = cfg->lookup("spectrum.makeBiorthonormal");
	}
    }
    else
      if (verboseCFG)
	std::cout << "Spectrum configuration section does not exist."
		  << std::endl;

    return;
}

  
/**
 * Sparse all configuration sections.
 * \param[in] cfgFileName Path to configuration file.
 */
void
readConfig(const char *cfgFileName, const bool verboseCFG)
{
  Config cfg;

  // Read the file. If there is an error, report it and exit.
  try {
    std::cout.precision(6);
    std::cout << "Reading config file " << cfgFileName << std::endl;
    cfg.readFile(cfgFileName);

    readGeneral(&cfg, verboseCFG);
    readModel(&cfg, verboseCFG);
    readSimulation(&cfg, verboseCFG);
    readContinuation(&cfg, verboseCFG);
    readSprinkle(&cfg, verboseCFG);
    readObservable(&cfg, verboseCFG);
    readGrid(&cfg, verboseCFG);
    readTransfer(&cfg, verboseCFG);
    readSpectrum(&cfg, verboseCFG);
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


  return;
}


    
/**
 * Free memory allocated during configuration.
 */
void
freeConfig()
{
  if (tauRng)
    gsl_vector_free(tauRng);
  if (gridLimitsLow)
    gsl_vector_free(gridLimitsLow);
  if (gridLimitsUp)
    gsl_vector_free(gridLimitsUp);
  if (nx)
    gsl_vector_uint_free(nx);
  if (embedding)
    gsl_vector_uint_free(embedding);
  if (components)
    gsl_vector_uint_free(components);
  if (seedRng)
    gsl_vector_uint_free(seedRng);
  if (initState)
    gsl_vector_free(initState);
  if (initCont)
    gsl_vector_free(initCont);
  if (minInitState)
    gsl_vector_free(minInitState);
  if (maxInitState)
    gsl_vector_free(maxInitState);

  return;
}
