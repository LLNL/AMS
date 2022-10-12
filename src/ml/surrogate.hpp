// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __AMS_SURROGATE_HPP__
#define __AMS_SURROGATE_HPP__

#include <string>

#ifdef __ENABLE_TORCH__
#include <torch/script.h>  // One-stop header.
#endif

#include "wf/data_handler.hpp"

#include "wf/debug.h"


//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class SurrogateModel
{

  static_assert(std::is_floating_point<TypeInValue>::value,
                "SurrogateModel supports floating-point values (floats, "
                "doubles, or long doubles) only!");

  using data_handler =
      ams::DataHandler<TypeInValue>;  // utils to handle float data

private:
  const std::string model_path;
  const bool is_cpu;


#ifdef __ENABLE_TORCH__
  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;
  c10::TensorOptions tensorOptions;


  // -------------------------------------------------------------------------
  // conversion to and from torch
  // -------------------------------------------------------------------------
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline at::Tensor arrayToTensor(long numRows,
                                  long numCols,
                                  TypeInValue** array)
  {
    c10::SmallVector<at::Tensor, 8> Tensors;
    for (int i = 0; i < numCols; i++) {
      Tensors.push_back(torch::from_blob((TypeInValue*)array[i],
                                         {numRows, 1},
                                         tensorOptions));
    }
    at::Tensor tensor = at::reshape(at::cat(Tensors, 1), {numRows, numCols});
    return tensor;
  }

#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline at::Tensor arrayToTensor(long numRows,
                                  long numCols,
                                  const TypeInValue** array)
  {
    c10::SmallVector<at::Tensor, 8> Tensors;
    for (int i = 0; i < numCols; i++) {
      Tensors.push_back(torch::from_blob((TypeInValue*)array[i],
                                         {numRows, 1},
                                         tensorOptions));
    }
    at::Tensor tensor = at::reshape(at::cat(Tensors, 1), {numRows, numCols});
    return tensor;
  }

#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void tensorToArray(at::Tensor tensor,
                            long numRows,
                            long numCols,
                            TypeInValue** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    tensor = tensor.transpose(1, 0);
    if (is_cpu) {
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        HtoHMemcpy(array[j], ptr, sizeof(TypeInValue) * numRows);
      }
    } else {
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        DtoDMemcpy(array[j], ptr, sizeof(TypeInValue) * numRows);
      }
    }
  }

  // -------------------------------------------------------------------------
  // loading a surrogate model!
  // -------------------------------------------------------------------------
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  void _load_torch(const std::string& model_path,
                   c10::Device&& device,
                   at::ScalarType dType)
  {
    try {
      module = torch::jit::load(model_path);
      module.to(device);
      module.to(dType);
      tensorOptions = torch::TensorOptions().dtype(dType).device(device);
    } catch (const c10::Error& e) {
      FATAL("Error loding torch model:%s", model_path.c_str())
    }
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    DBG(Surrogate, "Using model at double precision")
    _load_torch(model_path, torch::Device(device_name), torch::kFloat64);
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    DBG(Surrogate, "Using model at single precision")
    _load_torch(model_path, torch::Device(device_name), torch::kFloat32);
  }

  // -------------------------------------------------------------------------
  // evaluate a torch model
  // -------------------------------------------------------------------------
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void _evaluate(long num_elements,
                        long num_in,
                        size_t num_out,
                        const TypeInValue** inputs,
                        TypeInValue** outputs)
  {
    auto input = arrayToTensor(num_elements, num_in, inputs);
    at::Tensor output = module.forward({input}).toTensor();

    DBG(Surrogate, "Evaluate surrogate model (%ld, %ld) -> (%ld, %ld)",
        num_elements, num_in, num_elements, num_out);
    tensorToArray(output, num_elements, num_out, outputs);
  }

#else
  template <typename T>
#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
  }

#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void _evaluate(long num_elements,
                        long num_in,
                        size_t num_out,
                        TypeInValue** inputs,
                        TypeInValue** outputs)
  {
  }

#endif

  // -------------------------------------------------------------------------
  // public interface
  // -------------------------------------------------------------------------
public:
  SurrogateModel(const char* model_path, bool is_cpu = true)
      : model_path(model_path), is_cpu(is_cpu)
  {

    if (is_cpu)
      _load<TypeInValue>(model_path, "cpu");
    else
      _load<TypeInValue>(model_path, "cuda");
  }

#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void evaluate(long num_elements,
                       long num_in,
                       size_t num_out,
                       TypeInValue** inputs,
                       TypeInValue** outputs)
  {
    _evaluate(num_elements, num_in, num_out, inputs, outputs);
  }

#ifdef __ENABLE_PERFFLOWASPECT__
    __attribute__((annotate("@critical_path()")))
#endif
  inline void evaluate(long num_elements,
                       std::vector<const TypeInValue*> inputs,
                       std::vector<TypeInValue*> outputs)
  {
    _evaluate(num_elements,
              inputs.size(),
              outputs.size(),
              static_cast<const TypeInValue**>(inputs.data()),
              static_cast<TypeInValue**>(outputs.data()));
  }
};

#endif
