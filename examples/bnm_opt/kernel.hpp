#include <stddef.h>

#include "realtype.h"

#ifdef USE_AMS
#include <AMS.h>
#endif

class BinomialOptions
{
private:
  unsigned int batchSize;
  int rank;
  int worldSize;
  real *d_T, *d_R, *d_V;
  real *d_puByDf, *d_pdByDf, *d_vDt;
  real *d_S;
  real *d_X;
  real *d_CallValue;

#ifdef USE_AMS
  AMSCAbstrModel model;
  AMSExecutor wf;
#endif

public:
  BinomialOptions(unsigned int batchSize, int rank, int worldSize);
  void run(real *callValue,
           real *_S,
           real *_X,
           real *_R,
           real *_V,
           real *_T,
           size_t optN);

#ifdef USE_AMS
  static void AMSRun(void *cls, long numOptions, void **inputs, void **outputs);
#endif

  ~BinomialOptions();
};
