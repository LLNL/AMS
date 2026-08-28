#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif

#ifdef __AMS_ENABLE_CALIPER__
#include <caliper/cali_macros.h>
#endif

#include <execinfo.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../utils.hpp"
#include "AMS.h"
#include "macro.h"


template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  int sleep_msec;  // in milliseconds
  Problem(int ni, int no, int sleep_msec = 0)
      : num_inputs(ni), num_outputs(no), sleep_msec(sleep_msec)
  {
  }

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
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_msec));
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
    CALIPER(CALI_CXX_MARK_FUNCTION);

    CALIPER(CALI_CXX_MARK_LOOP_BEGIN(mainloop_id, "mainloop");)

    for (int i = 0; i < iterations; i++) {
      std::cout << "Iteration [" << i << "]\n";

      SmallVector<AMSTensor> input_tensors;
      SmallVector<AMSTensor> output_tensors;

      // Allocate Input memory
      for (int j = 0; j < num_inputs; j++) {
        DType* data = new DType[num_elements];
        DType* ptr = initialize_inputs(data, num_elements);
        input_tensors.push_back(AMSTensor::view(
            ptr,
            SmallVector<ams::AMSTensor::IntDimType>({num_elements, 1}),
            SmallVector<ams::AMSTensor::IntDimType>({1, 1}),
            resource));
      }

      // Allocate Output memory
      for (int j = 0; j < num_outputs; j++) {
        auto tmp = new DType[num_elements];
        output_tensors.push_back(AMSTensor::view(
            initialize_inputs(tmp, num_elements),
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
      CALIPER(CALI_CXX_MARK_LOOP_ITERATION(mainloop_id, i);)
    }
  }
};

int main(int argc, char** argv)
{
  // Number of ranks in this run
  int wS = 1;
  // My Local Id
  int rId = 0;
  // Level of Threading provided by MPI
  int provided = 0;

  installSignals();
  AMSInit();

  MPI_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &wS));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rId));

  // Deactivate output on cout for all ranks but 0
  if (rId != 0) {
    std::cout.setstate(std::ios::failbit);
  }

  std::string device_name = "cpu";
  std::string db_config = "";
  std::string db_type = "";
  std::string precision_opt = "double";

  int seed = 0;
  int num_elems = 1024;
  int num_inputs = 8;
  int num_outputs = 9;
  int num_iterations = 1;
  int sleep_msec = 0;
  bool verbose = false;

  // -------------------------------------------------------------------------
  // setup command line parser
  // -------------------------------------------------------------------------
  TestArgs args;
  args.AddOption(&device_name,
                 "-d",
                 "--device",
                 "Device config string (cpu or gpu)");

  // set precision
  args.AddOption(&precision_opt,
                 "-pr",
                 "--precision",
                 "Set precision (single or double)");

  // Sleeping time
  args.AddOption(&sleep_msec,
                 "-ms",
                 "--msleep",
                 "Sleep for x milliseconds for each iteration (default: 0)",
                 false);

  // data parameters
  args.AddOption(&num_elems,
                 "-e",
                 "--num-elems",
                 "Number of elements per iteration");

  args.AddOption(&num_inputs, "-di", "--dim-inputs", "Dimension of inputs");
  args.AddOption(&num_outputs, "-do", "--dim-outputs", "Dimension of outputs");
  args.AddOption(&num_iterations, "-i", "--num-iter", "Number of iterations");

  // random speed and packing
  args.AddOption(&seed, "-s", "--seed", "Seed for rand (default 0)", false);

  args.AddOption(&db_type,
                 "-dt",
                 "--dbtype",
                 "Configuration option of the different DB types:\n"
                 "\t 'hdf5': use HDF5 as a back end\n"
                 "\t 'rmq': use RabbitMQ as a back end\n");

  args.AddOption(
      &verbose, "-v", "--verbose", "Enable more verbose benchmark", false);

  // -------------------------------------------------------------------------
  // parse arguments
  // -------------------------------------------------------------------------
  args.Parse(argc, argv);
  if (!args.Good()) {
    args.PrintOptions();
    return -1;
  }

  if (rId == 0) {
    args.PrintUsage();
    std::cout << std::endl;
  }

  srand(seed + rId);

  AMSDType data_type = getDataType(precision_opt);
  AMSDBType dbType = getDBType(db_type);

  if (dbType == AMSDBType::AMS_NONE) {
    std::cerr << "Error: no DB backend specified with --dbtype\n";
    return -1;
  }

  const char* object_descr = std::getenv("AMS_OBJECTS");
  if (dbType == AMSDBType::AMS_RMQ && !object_descr) {
    std::cerr << "Error: RabbitMQ backend required to set env variable "
                 "AMS_OBJECTS\n";
    return -1;
  }

  // -------------------------------------------------------------------------
  // AMS allocators setup
  // -------------------------------------------------------------------------
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  const bool use_device = device_name == "gpu";
  if (use_device) {
#if defined(__AMS_ENABLE_CUDA__) || defined(__AMS_ENABLE_HIP__)
    resource = AMSResourceType::AMS_DEVICE;
#else
    std::cerr << "Error: Benchmark has not been compiled with Device support\n";
    return -1;
#endif
  }

  AMSCAbstrModel ams_model = AMSRegisterAbstractModel("bench_db_no_model",
                                                      0.5,
                                                      "",
                                                      "bench_db_no_model");


  std::cout << "Total elements across all " << wS
            << " ranks: " << wS * num_elems << "\n";
  std::cout << "Total elements per rank: " << num_elems << "\n";

  AMSExecutor wf = AMSCreateExecutor(ams_model, rId, wS);
  if (data_type == AMSDType::AMS_SINGLE) {
    Problem<float> prob(num_inputs, num_outputs, sleep_msec);
    prob.ams_run(wf, resource, num_iterations, num_elems);
  } else {
    Problem<double> prob(num_inputs, num_outputs, sleep_msec);
    prob.ams_run(wf, resource, num_iterations, num_elems);
  }

  MPI_CALL(MPI_Finalize());
  AMSFinalize();
  return 0;
}
