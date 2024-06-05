#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <ml/uq.hpp>
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <wf/basedb.hpp>
#include <wf/resource_manager.hpp>

#include "AMS.h"
#include "wf/debug.h"

void createUmpirePool(std::string parent_name, std::string pool_name)
{
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      pool_name, rm.getAllocator(parent_name));
}


AMSDType getDataType(char *d_type)
{
  AMSDType dType = AMSDType::AMS_DOUBLE;
  if (std::strcmp(d_type, "float") == 0) {
    dType = AMSDType::AMS_SINGLE;
  } else if (std::strcmp(d_type, "double") == 0) {
    dType = AMSDType::AMS_DOUBLE;
  } else {
    assert(false && "Unknown data type");
  }
  return dType;
}

template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  int multiplier;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no), multiplier(100) {}

  void run(long num_elements, DType **inputs, DType **outputs)
  {
    for (int i = 0; i < num_elements; i++) {
      DType sum = 0;
      for (int j = 0; j < num_inputs; j++) {
        sum += inputs[j][i];
      }

      for (int j = 0; j < num_outputs; j++) {
        outputs[j][i] = sum;
      }
    }
  }


  const DType *initialize_inputs(DType *inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor &wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    auto &rm = umpire::ResourceManager::getInstance();

    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      std::vector<const DType *> inputs;
      std::vector<DType *> outputs;

      // Allocate Input memory
      for (int j = 0; j < num_inputs; j++) {
        DType *data = new DType[elements];
        inputs.push_back(initialize_inputs(data, elements));
      }

      // Allocate Output memory
      for (int j = 0; j < num_outputs; j++) {
        outputs.push_back(new DType[elements]);
      }

      AMSExecute(wf,
                 (void *)this,
                 elements,
                 reinterpret_cast<const void **>(inputs.data()),
                 reinterpret_cast<void **>(outputs.data()),
                 inputs.size(),
                 outputs.size());

      for (int i = 0; i < num_outputs; i++) {
        delete outputs[i];
      }


      for (int i = 0; i < num_inputs; i++) {
        delete inputs[i];
      }
    }
  }
};

void callBackDouble(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the double model\n";
  static_cast<Problem<double> *>(cls)->run(elements,
                                           (double **)(inputs),
                                           (double **)(outputs));
}


void callBackSingle(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the single model\n";
  static_cast<Problem<float> *>(cls)->run(elements,
                                          (float **)(inputs),
                                          (float **)(outputs));
}


int main(int argc, char **argv)
{
  // Number of ranks in this run
  int wS = 1;
  // My Local Id
  int rId = 0;
  // Level of Threading provided by MPI
  int provided = 0;

  if (argc != 9) {
    std::cout << "Wrong cli\n";
    std::cout << argv[0]
              << " use_device(0|1) num_inputs num_outputs "
                 "data_type(float|double)"
                 "num_iterations avg_num_values 'model-name-1' 'model-name-2'";
    return -1;
  }


  int use_device = std::atoi(argv[1]);
  int num_inputs = std::atoi(argv[2]);
  int num_outputs = std::atoi(argv[3]);
  AMSDType data_type = getDataType(argv[4]);
  int num_iterations = std::atoi(argv[5]);
  int avg_elements = std::atoi(argv[6]);
  const char *model1 = argv[7];
  const char *model2 = argv[8];
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  srand(time(NULL));


  createUmpirePool("HOST", "TEST_HOST");
  AMSSetAllocator(AMSResourceType::AMS_HOST, "TEST_HOST");

  AMSCAbstrModel models[] = {AMSQueryModel(model1), AMSQueryModel(model2)};

  for (int i = 0; i < 2; i++) {
    if (data_type == AMSDType::AMS_SINGLE) {
      Problem<float> prob(num_inputs, num_outputs);
      AMSExecutor wf = AMSCreateExecutor(models[i],
                                         AMSDType::AMS_SINGLE,
                                         resource,
                                         (AMSPhysicFn)callBackSingle,
                                         0,
                                         1);

      prob.ams_run(wf, resource, num_iterations, avg_elements);
    } else {
      Problem<double> prob(num_inputs, num_outputs);
      AMSExecutor wf = AMSCreateExecutor(models[i],
                                         AMSDType::AMS_DOUBLE,
                                         resource,
                                         (AMSPhysicFn)callBackDouble,
                                         0,
                                         1);
      prob.ams_run(wf, resource, num_iterations, avg_elements);
    }
  }

  return 0;
}
