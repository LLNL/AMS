#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "AMS.h"

using namespace ams;

AMSDType getDataType(char* d_type)
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

  void run(long num_elements, const DType* const* inputs, DType** outputs)
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


  DType* initialize_inputs(DType* inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor& wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      SmallVector<AMSTensor> input_tensors;
      SmallVector<AMSTensor> output_tensors;

      // Allocate Input memory
      for (int j = 0; j < num_inputs; j++) {
        DType* data = new DType[elements];
        input_tensors.push_back(AMSTensor::view(
            initialize_inputs(data, elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      // Allocate Output memory
      for (int j = 0; j < num_outputs; j++) {
        auto tmp = new DType[elements];
        output_tensors.push_back(AMSTensor::view(
            initialize_inputs(tmp, elements),
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      DomainLambda OrigComputation =
          [&](const ams::SmallVector<ams::AMSTensor>& ams_ins,
              ams::SmallVector<ams::AMSTensor>& ams_inouts,
              ams::SmallVector<ams::AMSTensor>& ams_outs) {
            const DType* ins[num_inputs];
            DType* outs[num_outputs];
            if (num_inputs != ams_ins.size())
              throw std::runtime_error(
                  "Expecting dimensions of inputs to remain the same");
            else if (num_outputs != ams_outs.size())
              throw std::runtime_error(
                  "Expecting dimensions of outputs to remain the same");

            // Here I can use domain knowledge (inouts is empty)
            int num_elements = ams_ins[0].shape()[0];
            for (int i = 0; i < num_inputs; i++) {
              ins[i] = ams_ins[i].data<DType>();
              if (ams_ins[i].shape()[0] != num_elements)
                throw std::runtime_error(
                    "Expected tensors to have the same shape");
            }
            for (int i = 0; i < num_outputs; i++) {
              outs[i] = ams_outs[i].data<DType>();
              if (ams_outs[i].shape()[0] != num_elements)
                throw std::runtime_error(
                    "Expected tensors to have the same shape");
            }
            run(num_elements, ins, outs);
          };

      ams::SmallVector<AMSTensor> inouts;
      AMSExecute(wf, OrigComputation, input_tensors, inouts, output_tensors);

      for (int i = 0; i < input_tensors.size(); i++) {
        delete input_tensors[i].data<DType>();
      }


      for (int i = 0; i < output_tensors.size(); i++) {
        delete output_tensors[i].data<DType>();
      }
    }
  }
};

void callBackDouble(void* cls, long elements, void** inputs, void** outputs)
{
  std::cout << "Called the double model\n";
  static_cast<Problem<double>*>(cls)->run(elements,
                                          (double**)(inputs),
                                          (double**)(outputs));
}


void callBackSingle(void* cls, long elements, void** inputs, void** outputs)
{
  std::cout << "Called the single model\n";
  static_cast<Problem<float>*>(cls)->run(elements,
                                         (float**)(inputs),
                                         (float**)(outputs));
}


int main(int argc, char** argv)
{

  AMSInit();
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
  const char* model1 = argv[7];
  const char* model2 = argv[8];
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  srand(time(NULL));

  AMSCAbstrModel models[] = {AMSQueryModel(model1), AMSQueryModel(model2)};

  for (int i = 0; i < 2; i++) {
    AMSExecutor wf = AMSCreateExecutor(models[i], 0, 1);
    if (data_type == AMSDType::AMS_SINGLE) {
      Problem<float> prob(num_inputs, num_outputs);
      prob.ams_run(wf, resource, num_iterations, avg_elements);
    } else {
      Problem<double> prob(num_inputs, num_outputs);
      prob.ams_run(wf, resource, num_iterations, avg_elements);
    }
  }
  std::cout << "Finalize\n";
  AMSFinalize();
  std::cout << "Done with finalize\n";
  return 0;
}
