#include <stddef.h>

#include "realtype.h"

class BinomialOptions
{
private:
  unsigned int batchSize;
  real *d_T, *d_R, *d_V;
  real *d_puByDf, *d_pdByDf, *d_vDt;
  real *d_S;
  real *d_X;
  real *d_CallValue;

public:
  BinomialOptions(unsigned int batchSize);
  void run(real *callValue,
           real *_S,
           real *_X,
           real *_R,
           real *_V,
           real *_T,
           size_t optN);
  ~BinomialOptions();
};
