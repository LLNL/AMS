#include <mpi.h>
#include <AMS.h>

#include <cstring>
#include <iostream>
#include <random>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#include <wf/resource_manager.hpp>

#include "wf/redist_load.hpp"

#define SIZE (10)

void init(double *data, int elements, double value)
{
  for (int i = 0; i < elements; i++) {
    data[i] = value;
  }
}

void evaluate(double *data, double *src, int elements)
{
  ams::ResourceManager::copy(src, data, elements * sizeof(double));
}

int verify(double *data, double *src, int elements, int rId)
{
  return std::memcmp(data, src, elements * sizeof(double));
}

int main(int argc, char *argv[])
{
  using namespace ams;
  int device = std::atoi(argv[1]);
  MPI_Init(&argc, &argv);
  AMSSetupAllocator(AMSResourceType::HOST);
  AMSResourceType resource = AMSResourceType::HOST;
  AMSSetDefaultAllocator(AMSResourceType::HOST);
  int rId, wS;
  MPI_Comm_size(MPI_COMM_WORLD, &wS);
  MPI_Comm_rank(MPI_COMM_WORLD, &rId);
  srand(rId);
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.5, 0.3);
  srand(rId);
  double threshold;
  for ( int i = 0; i <= rId; i++){
    threshold = distribution(generator);
  }

  int computeElements = (threshold * SIZE); // / sizeof(double);

  double *srcData, *destData;
  double *srcHData = srcData =
      ResourceManager::allocate<double>(computeElements, AMSResourceType::HOST);
  double *destHData = destData =
      ResourceManager::allocate<double>(computeElements, AMSResourceType::HOST);

  init(srcHData, computeElements, static_cast<double>(rId));

  if (device == 1) {
    AMSSetupAllocator(AMSResourceType::DEVICE);
    AMSSetDefaultAllocator(AMSResourceType::DEVICE);
    resource = AMSResourceType::DEVICE;
    srcData = ResourceManager::allocate<double>(computeElements,
                                                AMSResourceType::DEVICE);
    destData = ResourceManager::allocate<double>(computeElements,
                                                 AMSResourceType::DEVICE);
  }

  std::vector<double *> inputs({srcData});
  std::vector<double *> outputs({destData});

  {

    std::cerr << "Resource is " << resource << "\n";
    AMSLoadBalancer<double> lBalancer(
        rId, wS, computeElements, MPI_COMM_WORLD, 1, 1, resource);
    lBalancer.scatterInputs(inputs, resource);
    double **lbInputs = lBalancer.inputs();
    double **lbOutputs = lBalancer.outputs();
    evaluate(*lbOutputs, *lbInputs, lBalancer.getBalancedSize());
    lBalancer.gatherOutputs(outputs, resource);
  }

  if (device == 1) {
    ResourceManager::copy(destData,
                          destHData,
                          computeElements * sizeof(double));
    ResourceManager::deallocate(destData, AMSResourceType::DEVICE);
    ResourceManager::deallocate(srcData, AMSResourceType::DEVICE);
  }

  int ret = verify(destHData, srcHData, computeElements, rId);

  ResourceManager::deallocate(destHData, AMSResourceType::HOST);
  ResourceManager::deallocate(srcHData, AMSResourceType::HOST);

  MPI_Finalize();
  return ret;
}
