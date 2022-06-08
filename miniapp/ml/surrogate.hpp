#ifndef _SURROGATE_EOS_HPP_
#define _SURROGATE_EOS_HPP_

#include <mfem/general/forall.hpp>
#include <string>

#ifdef __ENABLE_TORCH__
#include <torch/script.h> // One-stop header.
#endif

using namespace std;


#ifdef __ENABLE_CUDA__
#include <cuda_runtime.h>
inline void DtoDMemcpy(void *dest, void *src, size_t nBytes ){
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice);
}

inline void HtoHMemcpy(void *dest, void *src, size_t nBytes ){
  std::memcpy(dest, src, nBytes);
}

inline void HtoDMemcpy(void *dest, void *src, size_t nBytes ){
  cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice);
};

void DtoHMemcpy(void *dest, void *src, size_t nBytes ){
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost);
}
#else
inline void DtoDMemcpy(void *dest, void *src, size_t nBytes ){
  std::cerr<< "DtoD Memcpy Not Enabled" << std::endl;
  exit(-1);
}

inline void HtoHMemcpy(void *dest, void *src, size_t nBytes ){
  std::memcpy(dest, src, nBytes);
}

inline void HtoDMemcpy(void *dest, void *src, size_t nBytes ){
  std::cerr<< "HtoD Memcpy Not Enabled" << std::endl;
  exit(-1);
};

void DtoHMemcpy(void *dest, void *src, size_t nBytes ){
  std::cerr<< "DtoH Memcpy Not Enabled" << std::endl;
  exit(-1);
}
#endif

#include "app/eos.hpp"
#include "app/eos_idealgas.hpp"

#ifdef __ENABLE_TORCH__
//! An implementation for a surrogate model
class SurrogateModel {
  string model_path;
  bool is_cpu;
  const EOS *base_eos;
  torch::jit::script::Module module;
  c10::TensorOptions tensorOptions;
private:

  inline at::Tensor arrayToTensor(long numRows, long numCols, double **array){
    c10::SmallVector<at::Tensor, 8> Tensors;
    for (int i = 0; i < numCols; i++) {
      Tensors.push_back(torch::from_blob((double *)array[i], {numRows, 1}, tensorOptions));
    }
    at::Tensor tensor=
        at::reshape(at::cat(Tensors, 1), {numRows, numCols});
    return tensor;
  }

  inline void tensorToArray(at::Tensor tensor, long numRows, long numCols, double **array){
    // Transpose to get continuous memory and
    // perform single memcpy.
    tensor = tensor.transpose(1,0);
    if (is_cpu){
      std::cout << "Copying Host Outputs\n";
      for (long j = 0; j < numCols; j++){
        auto tmp = tensor[j].contiguous();
        double *ptr = tmp.data_ptr<double>();
        HtoHMemcpy(array[j], ptr, sizeof(double)*numRows);
      }
    }else{
      std::cout << "Copying Device Outputs\n";
      for (long j = 0; j < numCols; j++){
        auto tmp = tensor[j].contiguous();
        double *ptr = tmp.data_ptr<double>();
        DtoDMemcpy(array[j], ptr, sizeof(double)*numRows);
      }
    }
  }

public:
  SurrogateModel(EOS *_base_eos, const char *model_path, bool is_cpu=true)
      : model_path(model_path), base_eos(_base_eos), is_cpu(is_cpu) {
    try {
      std::cout << "File Name :" << model_path << "\n";
      if (is_cpu){
        module = torch::jit::load(model_path);
        tensorOptions = torch::TensorOptions().dtype(torch::kFloat64);
      }
      else{
        module = torch::jit::load(model_path);
        int id = 0;
        c10::Device device("cuda");
        module.to(device);
        tensorOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, id);
      }
    } catch (const c10::Error &e) {
      std::cerr << "error loading the model\n";
      exit(-1);
    }
  }

  void Eval(long num_elements, long num_in, size_t num_out, double **inputs,
            double **outputs) {
    auto input = arrayToTensor(num_elements, num_in, inputs);
    std::cout << "Shape I:" << input.sizes() << "\n";
    at::Tensor output = module.forward({input}).toTensor();
    std::cout << "Shape O:" << output.sizes() << "\n";
    tensorToArray(output, num_elements, num_out, outputs);
  }
};
#else
// At some point we will
// delete this version
// of the class.
class SurrogateModel {
  string model_path;
  bool is_cpu;
  const EOS *base_eos;

public:
  SurrogateModel(EOS *_base_eos, const char *model_path, bool is_cpu=true)
      : model_path(model_path), base_eos(_base_eos), is_cpu(is_cpu) {}

  void Eval(long num_elements, long num_in, size_t num_out, double **inputs,
            double **outputs) {
    base_eos->Eval(num_elements, inputs[0], inputs[1], outputs[0], outputs[1],
                   outputs[2], outputs[3]);
  }
};
#endif // __ENABLE_TORCH__
#endif
