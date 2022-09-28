#ifndef __AMS__
#define __AMS__

#ifdef __cplusplus

#ifdef __ENABLE_CALIPER__
#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#define CALIPER(stmt) stmt
#else
#define CALIPER(stmt)
#endif

extern "C" {
#endif

typedef void (*AMSPhysicFn) ( void *, long, const void * const * , void* const*);

typedef void* AMSExecutor;

typedef enum{
  Single = 0,
  Double
} AMSDType;

typedef enum {
  UNKNOWN = -1,
  HOST = 0,
  DEVICE = 1,
  RSEND
} AMSResourceType;

typedef enum{
 SinglePass = 0,
 Partition,
 Predicate
}AMSExecPolicy;

typedef struct ams_conf{
  const AMSExecPolicy ePolicy;
  const AMSDType dType;
  const AMSResourceType device;
  AMSPhysicFn cBack;
  char *SPath;
  char *UQPath;
  char *DBPath;
  double threshold;
}AMSConfig;

AMSExecutor AMSCreateExecutor(const AMSConfig config);
void AMSExecute(AMSExecutor executor, void* probDescr, const int numElements,
             const void **input_data, void **output_data,
             int inputDim, int outputDim);
void AMSDestroyExecutor(AMSExecutor executor);


const char *AMSGetAllocatorName(AMSResourceType device);
void AMSSetupAllocator(const AMSResourceType device);
void AMSResourceInfo();
int AMSGetLocationId(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
