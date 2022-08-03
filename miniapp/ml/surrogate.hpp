#ifndef __AMS_SURROGATE_HPP__
#define __AMS_SURROGATE_HPP__

#include <string>

#ifdef __ENABLE_TORCH__
#include <torch/script.h>  // One-stop header.
#endif

#include "app/eos.hpp"
#include "app/eos_idealgas.hpp"
#include "wf/data_handler.hpp"


//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class SurrogateModel {

    static_assert(
        std::is_floating_point<TypeInValue>::value,
        "SurrogateModel supports floating-point values (floats, doubles, or long doubles) only!");

    using data_handler = ams::DataHandler<TypeInValue>;  // utils to handle float data

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
    inline at::Tensor arrayToTensor(long numRows, long numCols, TypeInValue** array) {
        c10::SmallVector<at::Tensor, 8> Tensors;
        for (int i = 0; i < numCols; i++) {
            Tensors.push_back(
                torch::from_blob((TypeInValue*)array[i], {numRows, 1}, tensorOptions));
        }
        at::Tensor tensor = at::reshape(at::cat(Tensors, 1), {numRows, numCols});
        return tensor;
    }

    inline void tensorToArray(at::Tensor tensor, long numRows, long numCols,
                              TypeInValue** array) {
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
    void _load_torch(const std::string &model_path, c10::Device&& device, at::ScalarType dType) {
        try {
            module = torch::jit::load(model_path);
            module.to(device);
            module.to(dType);
            tensorOptions = torch::TensorOptions().dtype(dType).device(device);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading torch model\n";
            exit(-1);
        }
    }

    template <typename T, std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
    inline void _load(const std::string &model_path, const std::string &device_name) {
        std::cout << "Loading torch model ("<<model_path << ") at double precision\n";
        _load_torch(model_path, torch::Device(device_name), torch::kFloat64);
    }

    template <typename T, std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
    inline void _load(const std::string &model_path, const std::string &device_name) {
        std::cout << "Loading torch model ("<<model_path << ") at single precision\n";
        _load_torch(model_path, torch::Device(device_name), torch::kFloat32);
    }

    // -------------------------------------------------------------------------
    // evaluate a torch model
    // -------------------------------------------------------------------------
    inline void _evaluate(long num_elements, long num_in, size_t num_out,
                          TypeInValue** inputs, TypeInValue** outputs) {

        std::cout << "Evaluating surrogate model: ";
        fflush(stdout);

        auto input = arrayToTensor(num_elements, num_in, inputs);
        at::Tensor output = module.forward({input}).toTensor();

        std::cout << input.sizes() << " --> " << output.sizes() << "\n";
        tensorToArray(output, num_elements, num_out, outputs);
    }

#else
    template <typename T>
    inline void _load(const std::string &model_path, const std::string &device_name) {}

    inline void _evaluate(long num_elements, long num_in, size_t num_out,
                          TypeInValue** inputs, TypeInValue** outputs) {}

#endif

    // -------------------------------------------------------------------------
    // public interface
    // -------------------------------------------------------------------------
public:

    SurrogateModel(const char* model_path, bool is_cpu = true)
        : model_path(model_path), is_cpu(is_cpu) {

        if (is_cpu)   _load<TypeInValue>(model_path, "cpu");
        else          _load<TypeInValue>(model_path, "cuda");
    }

    inline void evaluate(long num_elements, long num_in, size_t num_out,
                         TypeInValue** inputs, TypeInValue** outputs) {
        _evaluate(num_elements, num_in, num_out, inputs, outputs);
    }

    inline void evaluate(long num_elements,
                         std::vector<TypeInValue*> inputs,
                         std::vector<TypeInValue*> outputs) {
        _evaluate(num_elements, inputs.size(), outputs.size(),
                  static_cast<TypeInValue**>(inputs.data()),
                  static_cast<TypeInValue**>(outputs.data()));
    }


#if 0
    // disabled by Harsh on July 08, 2022
    // --------- these are not needed, because
    // because once the application instantiates the surrogate model object
    // with TypeInValue, it should not be allowed to make queries in a different data type
    // this may cause inconsistencies later because based on the type, we decide to
    // cast the model into the appropriate type so we dont have to copy data back and forth
    // we still give the application the choice of data type, but take away the flexibility
    // to change the type to avoid abuse of data movement

    template <typename T = TypeInValue,
              std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
    SurrogateModel(const char* model_path, bool is_cpu = true)
        : model_path(model_path), is_cpu(is_cpu) {
#ifdef __ENABLE_TORCH__
        std::cout << "Using double precision models\n";
        if (is_cpu)
            loadModel(torch::kFloat64, torch::Device("cpu"));
        else
            loadModel(torch::kFloat64, torch::Device("cuda"));
#else
        std::cerr << "fatal: not built with PyTorch" << std::endl;
        exit(1);
#endif
    }

    template <typename T = TypeInValue,
              std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
    SurrogateModel(const char* model_path, bool is_cpu = true)
        : model_path(model_path), is_cpu(is_cpu) {
#ifdef __ENABLE_TORCH__
        std::cout << "Using single precision models\n";
        if (is_cpu)
            loadModel(torch::kFloat32, torch::Device("cpu"));
        else
            loadModel(torch::kFloat32, torch::Device("cuda"));
#else
        std::cerr << "fatal: not built with PyTorch" << std::endl;
        exit(1);
#endif
    }

    template <typename T, std::enable_if_t<std::is_same<TypeInValue, T>::value>* = nullptr>
    void Eval(long num_elements, long num_in, size_t num_out, T** inputs, T** outputs) {
        _evaluate(num_elements, num_in, num_out, inputs, outputs);
    }

    // similarly, this function is not needed
    // we require the application to pass in the same data type
    template <typename T, std::enable_if_t<!std::is_same<TypeInValue, T>::value>* = nullptr>
    void Eval(long num_elements, long num_in, size_t num_out, T** inputs, T** outputs) {

        std::vector<TypeInValue*> cinputs;
        std::vector<TypeInValue*> coutputs;
        for (int i = 0; i < num_in; i++) {
            cinputs.emplace_back(data_handler::cast_to_typevalue(num_elements, inputs[i]));
        }

        for (int i = 0; i < num_out; i++) {
            coutputs.emplace_back(new TypeInValue[num_elements]);
        }

        _evaluate(num_elements, num_in, num_out, static_cast<TypeInValue**>(cinputs.data()),
                  static_cast<TypeInValue**>(coutputs.data()));

        for (int i = 0; i < num_out; i++) {
            delete[] cinputs[i];
        }

        for (int i = 0; i < num_out; i++) {
            ams::DataHandler<T>::cast_from_typevalue(num_elements, outputs[i], coutputs[i]);
            delete[] coutputs[i];
        }
    }

    template <typename T>
    void Eval(long num_elements, std::vector<T*> inputs, std::vector<T*> outputs) {
        Eval(num_elements, inputs.size(), outputs.size(), static_cast<T**>(inputs.data()),
             static_cast<T**>(outputs.data()));
    }
#endif
};

#endif
