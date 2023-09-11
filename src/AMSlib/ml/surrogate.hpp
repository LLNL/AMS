/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_SURROGATE_HPP__
#define __AMS_SURROGATE_HPP__

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef __ENABLE_TORCH__
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
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
  AMSResourceType model_resource;
  const bool _is_DeltaUQ;

#ifdef __ENABLE_TORCH__
  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;
  c10::TensorOptions tensorOptions;


  // -------------------------------------------------------------------------
  // conversion to and from torch
  // -------------------------------------------------------------------------
  PERFFASPECT()
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

  PERFFASPECT()
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

  PERFFASPECT()
  inline void tensorToArray(at::Tensor tensor,
                            long numRows,
                            long numCols,
                            TypeInValue** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    tensor = tensor.transpose(1, 0);
    if (model_resource == AMSResourceType::HOST) {
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

  PERFFASPECT()
  inline void tensorToHostArray(at::Tensor tensor,
                                long numRows,
                                long numCols,
                                TypeInValue** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    tensor = tensor.transpose(1, 0);
    if (model_resource == AMSResourceType::HOST) {
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        HtoHMemcpy(array[j], ptr, sizeof(TypeInValue) * numRows);
      }
    } else {
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        DtoHMemcpy(array[j], ptr, sizeof(TypeInValue) * numRows);
      }
    }
  }

  // -------------------------------------------------------------------------
  // loading a surrogate model!
  // -------------------------------------------------------------------------
  PERFFASPECT()
  void _load_torch(const std::string& model_path,
                   c10::Device&& device,
                   at::ScalarType dType)
  {
    try {
      module = torch::jit::load(model_path);
      module.to(device);
      module.to(dType);
      tensorOptions =
          torch::TensorOptions().dtype(dType).device(device).requires_grad(
              false);
    } catch (const c10::Error& e) {
      FATAL("Error loding torch model:%s", model_path.c_str())
    }
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
  PERFFASPECT()
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    DBG(Surrogate, "Using model(%s) at double precision at device %s", model_path.c_str(), device_name.c_str());
    _load_torch(model_path, torch::Device(device_name), torch::kFloat64);
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
  PERFFASPECT()
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    DBG(Surrogate, "Using model (%s) at single precision at device %s", model_path.c_str(), device_name.c_str())
    _load_torch(model_path, torch::Device(device_name), torch::kFloat32);
  }

  // -------------------------------------------------------------------------
  // evaluate a torch model
  // -------------------------------------------------------------------------
  PERFFASPECT()
  inline void _evaluate(long num_elements,
                        size_t num_in,
                        size_t num_out,
                        const TypeInValue** inputs,
                        TypeInValue** outputs,
                        TypeInValue** outputs_stdev)
  {
    //torch::NoGradGuard no_grad;
    c10::InferenceMode guard(true);
    auto input = arrayToTensor(num_elements, num_in, inputs);
    input.set_requires_grad(false);
    if (_is_DeltaUQ) {
      assert(outputs_stdev && "Expected non-null outputs_stdev");
      // The deltauq surrogate returns a tuple of (outputs, outputs_stdev)
      auto output_tuple = module.forward({input}).toTuple();
      at::Tensor output_mean_tensor =
          output_tuple->elements()[0].toTensor().detach();
      at::Tensor output_stdev_tensor =
          output_tuple->elements()[1].toTensor().detach();
      tensorToArray(output_mean_tensor, num_elements, num_out, outputs);
      tensorToHostArray(output_stdev_tensor,
                        num_elements,
                        num_out,
                        outputs_stdev);
    } else {
      at::Tensor output = module.forward({input}).toTensor().detach();
      tensorToArray(output, num_elements, num_out, outputs);
    }

    if (is_device()) {
      deviceCheckErrors(__FILE__, __LINE__);
    }

    DBG(Surrogate,
        "Evaluate surrogate model (%ld, %ld) -> (%ld, %ld)",
        num_elements,
        num_in,
        num_elements,
        num_out);
  }

#else
  template <typename T>
  PERFFASPECT()
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
  }

  PERFFASPECT()
  inline void _evaluate(long num_elements,
                        long num_in,
                        size_t num_out,
                        const TypeInValue** inputs,
                        TypeInValue** outputs,
                        TypeInValue** outputs_stdev)
  {
  }

#endif

  SurrogateModel(const char* model_path,
                 AMSResourceType resource = AMSResourceType::HOST,
                 bool is_DeltaUQ = false)
      : model_path(model_path),
        model_resource(resource),
        _is_DeltaUQ(is_DeltaUQ)
  {
    if (resource != AMSResourceType::DEVICE)
      _load<TypeInValue>(model_path, "cpu");
    else
      _load<TypeInValue>(model_path, "cuda");
  }

protected:
  template <typename T,
            std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
  static bool same_type(bool is_double)
  {
    return !is_double;
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
  static bool same_type(bool is_double)
  {
    return is_double;
  }

  static std::unordered_map<std::string,
                            std::shared_ptr<SurrogateModel<TypeInValue>>>
      instances;

public:
  // -------------------------------------------------------------------------
  // public interface
  // -------------------------------------------------------------------------

  static std::shared_ptr<SurrogateModel<TypeInValue>> getInstance(
      const char* model_path,
      AMSResourceType resource = AMSResourceType::HOST,
      bool is_DeltaUQ = false)
  {
    auto model =
        SurrogateModel<TypeInValue>::instances.find(std::string(model_path));
    if (model != instances.end()) {
      // Model Found
      auto torch_model = model->second;
      if (resource != torch_model->model_resource)
        throw std::runtime_error(
            "Currently we are not supporting loading the same model file on "
            "different devices.");

      if (is_DeltaUQ != torch_model->is_DeltaUQ())
        THROW(std::runtime_error, "Loaded model instance is not DeltaUQ");

      if (!same_type<TypeInValue>(torch_model->is_double()))
        throw std::runtime_error(
            "Requesting model loading of different data types.");

      DBG(Surrogate,
          "Returning existing model represented under (%s)",
          model_path);
      return torch_model;
    }

    // Model does not exist. We need to create one
    DBG(Surrogate, "Generating new model under (%s)", model_path);
    std::shared_ptr<SurrogateModel<TypeInValue>> torch_model =
        std::shared_ptr<SurrogateModel<TypeInValue>>(
            new SurrogateModel<TypeInValue>(model_path, resource, is_DeltaUQ));
    instances.insert(std::make_pair(std::string(model_path), torch_model));
    return torch_model;
  };

  ~SurrogateModel()
  {
    DBG(Surrogate, "Destroying surrogate model at %s", model_path.c_str());
  }


  PERFFASPECT()
  inline void evaluate(long num_elements,
                       size_t num_in,
                       size_t num_out,
                       const TypeInValue** inputs,
                       TypeInValue** outputs,
                       TypeInValue** outputs_stdev = nullptr)
  {
    _evaluate(num_elements, num_in, num_out, inputs, outputs, outputs_stdev);
  }

  PERFFASPECT()
  inline void evaluate(long num_elements,
                       std::vector<const TypeInValue*> inputs,
                       std::vector<TypeInValue*> outputs,
                       std::vector<TypeInValue*> outputs_stdev)
  {
    _evaluate(num_elements,
              inputs.size(),
              outputs.size(),
              static_cast<const TypeInValue**>(inputs.data()),
              static_cast<TypeInValue**>(outputs.data()),
              static_cast<TypeInValue**>(outputs_stdev.data()));
  }

  PERFFASPECT()
  inline void evaluate(long num_elements,
                       std::vector<const TypeInValue*> inputs,
                       std::vector<TypeInValue*> outputs)
  {
    _evaluate(num_elements,
              inputs.size(),
              outputs.size(),
              static_cast<const TypeInValue**>(inputs.data()),
              static_cast<TypeInValue**>(outputs.data()),
              nullptr);
  }

#ifdef __ENABLE_TORCH__
  bool is_double() { return (tensorOptions.dtype() == torch::kFloat64); }
#else
  bool is_double()
  {
    if (typeid(TypeInValue) == typeid(double)) return true;
    return false;
  }

#endif

  inline bool is_device() const
  {
#ifdef __ENABLE_TORCH__
    return model_resource == AMSResourceType::DEVICE;
#else
    return false;
#endif
  }

  bool is_DeltaUQ() { return _is_DeltaUQ; }
};

template <typename T>
std::unordered_map<std::string, std::shared_ptr<SurrogateModel<T>>>
    SurrogateModel<T>::instances;

#endif
