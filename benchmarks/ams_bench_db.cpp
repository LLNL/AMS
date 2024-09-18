#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif

#ifdef __AMS_ENABLE_ADIAK__
#include <adiak.hpp>
#endif

#ifdef __AMS_ENABLE_CALIPER__
#include <caliper/cali_macros.h>
#endif

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unistd.h>

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include <mfem.hpp>

#include "AMS.h"

void createUmpirePool(const std::string& parent_name, const std::string& pool_name)
{
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      pool_name, rm.getAllocator(parent_name));
}

AMSDType getDataType(const char* d_type)
{
  AMSDType dType = AMSDType::AMS_DOUBLE;
  if (std::strcmp(d_type, "float") == 0) {
    dType = AMSDType::AMS_SINGLE;
  } else if (d_type ==  "double") {
    dType = AMSDType::AMS_DOUBLE;
  } else {
    assert(false && "Unknown data type (must be 'float' or 'double')");
  }
  return dType;
}

AMSDBType getDBType(const char* db_type)
{
  AMSDBType dbType = AMSDBType::AMS_NONE;
  if (std::strcmp(db_type, "csv") == 0) {
    dbType = AMSDBType::AMS_CSV;
  } else if (std::strcmp(db_type, "hdf5") == 0) {
    dbType = AMSDBType::AMS_HDF5;
  } else if (std::strcmp(db_type, "rmq") == 0) {
    dbType = AMSDBType::AMS_RMQ;
  }
  return dbType;
}

template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no) {}

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


/*
To move to CUDA
      FPTypeValue *pPtr =
          rm.allocate<FPTypeValue>(num_elements, AMSResourceType::AMS_HOST);
      rm.copy(outputs[i], AMS_DEVICE, pPtr, AMS_HOST, num_elements);
*/

  void ams_run(AMSExecutor &wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    CALIPER(CALI_CXX_MARK_FUNCTION;)
    auto &rm = umpire::ResourceManager::getInstance();

    CALIPER(CALI_CXX_MARK_LOOP_BEGIN(mainloop_id, "mainloop");)

    for (int i = 0; i < iterations; i++) {
      CALIPER(CALI_CXX_MARK_LOOP_ITERATION(mainloop_id, i);)
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
        delete[] outputs[i];
        outputs[i] = nullptr;
      }
      for (int i = 0; i < num_inputs; i++) {
        delete[] inputs[i];
        inputs[i] = nullptr;
      }
    }
    CALIPER(CALI_CXX_MARK_LOOP_END(mainloop_id);)
  }
};

void callBackDouble(void *cls, long elements, void **inputs, void **outputs)
{
  // std::cout << "Called the double precision model\n";
  static_cast<Problem<double> *>(cls)->run(elements,
                                           (double **)(inputs),
                                           (double **)(outputs));
}


void callBackSingle(void *cls, long elements, void **inputs, void **outputs)
{
  // std::cout << "Called the single precision model\n";
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
  MPI_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &wS));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rId));
  if (rId != 0) {
    std::cout.setstate(std::ios::failbit);
  }

  const char *device_name = "cpu";
  const char *db_config = "";
  const char *db_type = "";
  const char *precision_opt = "double";

  int seed = 0;
  int num_elems = 1024;
  int num_inputs = 8;
  int num_outputs = 9;
  int num_iterations = 1;
  bool verbose = false;
  bool reqDB = false;
#ifdef __ENABLE_DB__
  reqDB = true;
#endif

  // -------------------------------------------------------------------------
  // setup command line parser
  // -------------------------------------------------------------------------
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&device_name, "-d", "--device", "Device config string (cpu or cuda)");

  // set precision
  args.AddOption(&precision_opt,
                 "-pr",
                 "--precision",
                 "Set precision (single or double)");

  // data parameters
  args.AddOption(&num_elems, "-e", "--num-elems", "Number of elements per iteration");
  args.AddOption(&num_inputs, "-di", "--dim-inputs", "Dimension of inputs");
  args.AddOption(&num_outputs, "-do", "--dim-outputs", "Dimension of outputs");
  args.AddOption(&num_iterations, "-i", "--num-iter", "Number of iterations");

  // random speed and packing
  args.AddOption(&seed, "-s", "--seed", "Seed for rand (default 0)");

  args.AddOption(&db_config,
                 "-db",
                 "--dbconfig",
                 "Path to directory where applications will store their data (for CSV/HDF5)",
                 reqDB);

  args.AddOption(&db_type,
                 "-dt",
                 "--dbtype",
                 "Configuration option of the different DB types:\n"
                 "\t 'csv': use CSV as a back end\n"
                 "\t 'hdf5': use HDF5 as a back end\n"
                 "\t 'rmq': use RabbitMQ as a back end\n");

  args.AddOption(
      &verbose, "-v", "--verbose", "-qu", "--quiet", "Enable more verbose benchmark");

  // -------------------------------------------------------------------------
  // parse arguments
  // -------------------------------------------------------------------------
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return -1;
  }

  if (rId == 0) {
    args.PrintOptions(std::cout);
    std::cout << std::endl;
  }

  srand(seed + rId);

  // -------------------------------------------------------------------------
  // Adiak
  // -------------------------------------------------------------------------
#ifdef __AMS_ENABLE_ADIAK__
  // add adiak init here
  adiak::init(NULL);

  // replace with adiak::collect_all(); once adiak v0.4.0
  adiak::uid();
  adiak::launchdate();
  adiak::launchday();
  adiak::executable();
  adiak::executablepath();
  adiak::workdir();
  adiak::libraries();
  adiak::cmdline();
  adiak::hostname();
  adiak::clustername();
  adiak::walltime();
  adiak::systime();
  adiak::cputime();
  adiak::jobsize();
  adiak::hostlist();
  adiak::numhosts();
  adiak::value("compiler", std::string("@RAJAPERF_COMPILER@"));
#endif

  AMSDType data_type = getDataType(precision_opt);
  AMSDBType dbType = getDBType(db_type);

  if (dbType == AMSDBType::AMS_NONE) {
    std::cerr << "Error: no DB backend specified with --dbtype\n";
    return -1;
  }

  const char *object_descr = std::getenv("AMS_OBJECTS");
  if (dbType == AMSDBType::AMS_RMQ && !object_descr) {
    std::cerr << "Error: RabbitMQ backend required to set env variable AMS_OBJECTS\n";
    return -1;
  }

  if (dbType != AMSDBType::AMS_RMQ) {
    AMSConfigureFSDatabase(dbType, db_config);
  }

  // -------------------------------------------------------------------------
  // AMS allocators setup
  // -------------------------------------------------------------------------
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  const bool use_device = std::strcmp(device_name, "cpu") != 0;
  if (use_device) {
#ifdef __ENABLE_CUDA__
    resource = AMSResourceType::AMS_DEVICE;
#else
    std::cerr << "Error: Benchmark has not been compiled with CUDA support\n";
    return -1;
#endif
  }

  AMSCAbstrModel ams_model = AMSRegisterAbstractModel("bench_db_no_model",
                                                        AMSUQPolicy::AMS_RANDOM,
                                                        0.5,
                                                        "",
                                                        "",
                                                        "bench_db_no_model",
                                                        1);

  std::cout << "Total elements across all " << wS << " ranks: " << wS * num_elems
            << " (Weak Scaling)\n";
  std::cout << "Total elements per rank: " << num_elems << "\n";

  if (data_type == AMSDType::AMS_SINGLE) {
    Problem<float> prob(num_inputs, num_outputs);
    AMSExecutor wf = AMSCreateExecutor(ams_model,
                                        AMSDType::AMS_SINGLE,
                                        resource,
                                        (AMSPhysicFn)callBackSingle,
                                        rId,
                                        wS);

    prob.ams_run(wf, resource, num_iterations, num_elems);
  } else {
    Problem<double> prob(num_inputs, num_outputs);
    AMSExecutor wf = AMSCreateExecutor(ams_model,
                                        AMSDType::AMS_DOUBLE,
                                        resource,
                                        (AMSPhysicFn)callBackDouble,
                                        rId,
                                        wS);
    prob.ams_run(wf, resource, num_iterations, num_elems);
  }

#ifdef __AMS_ENABLE_ADIAK__
  adiak::fini()
#endif

  return 0;
}
