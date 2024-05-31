/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_SURROGATE_HPP__
#define __AMS_SURROGATE_HPP__

#include <experimental/filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "AMS.h"
#include "wf/cuda/utilities.cuh"

#ifdef __ENABLE_TORCH__
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/cuda.h>
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
    CALIPER(CALI_MARK_BEGIN("ARRAY_BLOB");)
    for (int i = 0; i < numCols; i++) {
      Tensors.push_back(torch::from_blob((TypeInValue*)array[i],
                                         {numRows, 1},
                                         tensorOptions));
    }
    CALIPER(CALI_MARK_END("ARRAY_BLOB");)

    CALIPER(CALI_MARK_BEGIN("ARRAY_RESHAPE");)
    at::Tensor tensor = at::reshape(at::cat(Tensors, 1), {numRows, numCols});
    CALIPER(CALI_MARK_END("ARRAY_RESHAPE");)
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
    if (model_resource == AMSResourceType::AMS_HOST) {
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
    if (model_resource == AMSResourceType::AMS_HOST) {
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
    DBG(Surrogate, "Using model at double precision");
    _load_torch(model_path, torch::Device(device_name), torch::kFloat64);
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
  PERFFASPECT()
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    DBG(Surrogate, "Using model at single precision");
    _load_torch(model_path, torch::Device(device_name), torch::kFloat32);
  }

  // -------------------------------------------------------------------------
  // compute delta uq predicates
  // -------------------------------------------------------------------------
  void computeDeltaUQPredicates(AMSUQPolicy uq_policy,
                                const TypeInValue* __restrict__ outputs_stdev,
                                bool* __restrict__ predicates,
                                const size_t nrows,
                                const size_t ncols,
                                const double threshold)
  {
    auto computeDeltaUQMeanPredicatesHost = [&]() {
      for (size_t i = 0; i < nrows; ++i) {
        double mean = 0.0;
        for (size_t j = 0; j < ncols; ++j)
          mean += outputs_stdev[j + i * ncols];
        mean /= ncols;

        predicates[i] = (mean < threshold);
      }
    };

    auto computeDeltaUQMaxPredicatesHost = [&]() {
      for (size_t i = 0; i < nrows; ++i) {
        predicates[i] = true;
        for (size_t j = 0; j < ncols; ++j)
          if (outputs_stdev[j + i * ncols] >= threshold) {
            predicates[i] = false;
            break;
          }
      }
    };

    if (uq_policy == AMSUQPolicy::AMS_DELTAUQ_MEAN) {
      if (model_resource == AMSResourceType::AMS_DEVICE){
#ifdef __ENABLE_CUDA__
        DBG(Surrogate, "Compute mean delta uq predicates on device\n");
        constexpr int block_size = 256;
        int grid_size = divup(nrows, block_size);
        computeDeltaUQMeanPredicatesKernel<<<grid_size, block_size>>>(
            outputs_stdev, predicates, nrows, ncols, threshold);
        // TODO: use combined routine when it lands.
        cudaDeviceSynchronize();
        CUDACHECKERROR();
#else
        THROW(std::runtime_error,
              "Expected CUDA is enabled when model data are on DEVICE");
#endif
      }
      else {
        DBG(Surrogate, "Compute mean delta uq predicates on host\n");
        computeDeltaUQMeanPredicatesHost();
      }
    } else if (uq_policy == AMSUQPolicy::AMS_DELTAUQ_MAX) {
      if (model_resource == AMSResourceType::AMS_DEVICE){
#ifdef __ENABLE_CUDA__
        DBG(Surrogate, "Compute max delta uq predicates on device\n");
        constexpr int block_size = 256;
        int grid_size = divup(nrows, block_size);
        computeDeltaUQMaxPredicatesKernel<<<grid_size, block_size>>>(
            outputs_stdev, predicates, nrows, ncols, threshold);
        // TODO: use combined routine when it lands.
        cudaDeviceSynchronize();
        CUDACHECKERROR();
#else
        THROW(std::runtime_error,
              "Expected CUDA is enabled when model data are on DEVICE");
#endif
      }
      else {
        DBG(Surrogate, "Compute max delta uq predicates on host\n");
        computeDeltaUQMaxPredicatesHost();
      }
    } else
      THROW(std::runtime_error,
            "Invalid uq_policy to compute delta uq predicates");
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
                        AMSUQPolicy uq_policy,
                        bool* predicates,
                        double threshold)
  {
    //torch::NoGradGuard no_grad;
    c10::InferenceMode guard(true);
    CALIPER(CALI_MARK_BEGIN("ARRAY_TO_TENSOR");)
    auto input = arrayToTensor(num_elements, num_in, inputs);
    CALIPER(CALI_MARK_END("ARRAY_TO_TENSOR");)

    input.set_requires_grad(false);
    if (_is_DeltaUQ) {
      // The deltauq surrogate returns a tuple of (outputs, outputs_stdev)
      CALIPER(CALI_MARK_BEGIN("SURROGATE-EVAL");)
      auto output_tuple = module.forward({input}).toTuple();
      CALIPER(CALI_MARK_END("SURROGATE-EVAL");)

      at::Tensor output_mean_tensor =
          output_tuple->elements()[0].toTensor().detach();
      at::Tensor output_stdev_tensor =
          output_tuple->elements()[1].toTensor().detach();
      CALIPER(CALI_MARK_BEGIN("TENSOR_TO_ARRAY");)

      computeDeltaUQPredicates(uq_policy,
                               output_stdev_tensor.data_ptr<TypeInValue>(),
                               predicates,
                               num_elements,
                               num_out,
                               threshold);
      tensorToArray(output_mean_tensor, num_elements, num_out, outputs);
      CALIPER(CALI_MARK_END("TENSOR_TO_ARRAY");)
    } else {
      CALIPER(CALI_MARK_BEGIN("SURROGATE-EVAL");)
      at::Tensor output = module.forward({input}).toTensor().detach();
      CALIPER(CALI_MARK_END("SURROGATE-EVAL");)

      CALIPER(CALI_MARK_BEGIN("TENSOR_TO_ARRAY");)
      tensorToArray(output, num_elements, num_out, outputs);
      CALIPER(CALI_MARK_END("TENSOR_TO_ARRAY");)
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
                        AMSUQPolicy uq_policy,
                        bool* predicates,
                        double threshold)
  {
  }

#endif

  SurrogateModel(std::string& model_path,
                 AMSResourceType resource = AMSResourceType::AMS_HOST,
                 bool is_DeltaUQ = false)
      : model_path(model_path),
        model_resource(resource),
        _is_DeltaUQ(is_DeltaUQ)
  {

    std::experimental::filesystem::path Path(model_path);
    std::error_code ec;

    if (!std::experimental::filesystem::exists(Path, ec)) {
      FATAL(Surrogate,
            "Path to Surrogate Model (%s) Does not exist",
            model_path.c_str())
    }

    if (resource != AMSResourceType::AMS_DEVICE)
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
      std::string& model_path,
      AMSResourceType resource = AMSResourceType::AMS_HOST,
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
    DBG(Surrogate, "Generating new model under (%s)", model_path.c_str());
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
                       AMSUQPolicy uq_policy = AMSUQPolicy::AMS_UQ_BEGIN,
                       bool* predicates = nullptr,
                       double threshold = 0.0)
  {
    _evaluate(num_elements,
              num_in,
              num_out,
              inputs,
              outputs,
              uq_policy,
              predicates,
              threshold);
  }

  PERFFASPECT()
  inline void evaluate(long num_elements,
                       std::vector<const TypeInValue*> inputs,
                       std::vector<TypeInValue*> outputs,
                       AMSUQPolicy uq_policy,
                       bool* predicates,
                       double threshold)
  {
    _evaluate(num_elements,
              inputs.size(),
              outputs.size(),
              static_cast<const TypeInValue**>(inputs.data()),
              static_cast<TypeInValue**>(outputs.data()),
              uq_policy,
              predicates,
              threshold);
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
              AMSUQPolicy::AMS_UQ_BEGIN,
              nullptr,
              0.0);
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
    return model_resource == AMSResourceType::AMS_DEVICE;
#else
    return false;
#endif
  }

  bool is_DeltaUQ() { return _is_DeltaUQ; }

  void update(const std::string& new_path)
  {
    /* This function updates the underlying torch model,
     * with a new one pointed at location modelPath. The previous
     * one is destructed automatically.
     *
     * TODO: I decided to not update the model path on the ``instances''
     * map. As we currently expect this change will be agnostic to the application
     * user. But, in any case we should keep track of which model has been used at which
     * invocation. This is currently not done.
     */
    if (model_resource != AMSResourceType::AMS_DEVICE)
      _load<TypeInValue>(new_path, "cpu");
    else
      _load<TypeInValue>(new_path, "cuda");
  }

  AMSResourceType getModelResource() const { return model_resource; }
};

template <typename T>
std::unordered_map<std::string, std::shared_ptr<SurrogateModel<T>>>
    SurrogateModel<T>::instances;

#endif
