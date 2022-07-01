#ifndef __AMS_UTILS_MFEM_HPP__
#define __AMS_UTILS_MFEM_HPP__

#include <array>
#include <iostream>

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"


#define RESHAPE_TENSOR(m, op) mfem::Reshape(m.op(), m.SizeI(), m.SizeJ(), m.SizeK())


template<typename T>
static void
print_tensor_array(const std::string &label,
                   const T *values,
                   const std::array<int,3> &sz) {

    const int K = sz[0], J = sz[1], I = sz[2];
    if (K == 1) {
        std::cout << "--> printing ["<<J<<" x "<<I<<"] tensor \""<<label<<"\"\n";
        for (int j = 0; j < J; ++j) {
        for (int i = 0; i < I; ++i) {
            int idx = i + I*j;
            std::cout << label << "["<<j<<","<<i<<"] = " << idx << " = " << values[idx] << std::endl;
        }}
    }
    else {
        std::cout << "--> printing ["<<K<<" x "<<J<<" x "<<I<<"] tensor \""<<label<<"\"\n";
        for (int k = 0; k < K; ++k) {
        for (int j = 0; j < J; ++j) {
        for (int i = 0; i < I; ++i) {
            int idx = i + I*(j + k*J);
            std::cout << label << "["<<k<<", "<<j<<","<<i<<"] = " << values[idx] << std::endl;
        }}}
    }
}


template<typename T>
static void
print_array(const std::string &label,
            const mfem::Array<T> &values) {

    std::cout << "--> printing [sz = "<<values.Size()<<"] array \""<<label<<"\" = " << values <<"\n";
    for (int i = 0; i < values.Size(); ++i) {
        std::cout << label << "["<<i<<"] = " << values[i] << std::endl;
    }
}


void print_dense_tensor(const std::string &label,
                        const mfem::DenseTensor &values) {

    const int I = values.SizeI(), J = values.SizeJ(), K = values.SizeK();
    std::cerr << "--> printing ["<<I<<" x "<<J<<" x "<<K<<"] dense_tensor \""<<label<<"\"" << std::endl;// = " << values << "\n";

    for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
    for (int i = 0; i < I; ++i) {
        std::cerr << label<<"["<<i<<","<<j<<", "<<k<<"] = " << values(i,j,k) << std::endl;
    }}}
}

void print_dense_tensor(const std::string &label,
                        const mfem::DenseTensor &values,
                        const mfem::DeviceTensor<2, bool>& filter) {

    const int I = values.SizeI(), J = values.SizeJ(), K = values.SizeK();
    std::cout << "--> printing ["<<I<<" x "<<J<<" x "<<K<<"] dense_tensor \""<<label<<"\"\n";// = " << values << "\n";

    for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
        if (!filter(j,k)) {
            continue;
        }
        for (int i = 0; i < I; ++i) {
            std::cout << label<<"["<<i<<","<<j<<", "<<k<<"] = " << values(i,j,k) <<"\n";
        }
    }}
}

void print_dense_tensor(const std::string &label,
                        const mfem::DenseTensor &values,
                        const bool *filter) {

    const int I = values.SizeI(), J = values.SizeJ(), K = values.SizeK();
    std::cout << "--> printing ["<<I<<" x "<<J<<" x "<<K<<"] dense_tensor \""<<label<<"\"\n";// = " << values << "\n";

    for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
        if (!filter[j+k*J])
            continue;
        for (int i = 0; i < I; ++i) {
            std::cout << label<<"["<<i<<","<<j<<", "<<k<<"] = " << values(i,j,k) <<"\n";
        }
    }}
}

template<typename T>
static void
print_device_tensor(const std::string &label,
                    const mfem::DeviceTensor<2,T> &values,
                    const std::array<int,2> &sz) {

    const int I = sz[0], J = sz[1];
    std::cout << "--> printing ["<<I<<" x "<<J<<"] device_tensor \""<<label<<"\" = " << values << "\n";
    for (int i = 0; i < sz[0]; ++i) {
    for (int j = 0; j < sz[1]; ++j) {
        std::cout << label << "["<<i<<"]["<<j<<"] = " << values(i,j) << std::endl;
    }}
}


template<typename T>
static void
print_device_tensor(const std::string &label,
                    const mfem::DeviceTensor<3,T> &values,
                    const std::array<int,3> &sz) {

    const int I = sz[0], J = sz[1], K = sz[2];
    std::cout << "--> printing ["<<I<<" x "<<J<<" x "<<K<<"] device_tensor \""<<label<<"\"\n";
    for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
    for (int k = 0; k < K; ++k) {
        std::cout << label << "["<<i<<"]["<<j<<"]["<<k<<"] = " << values(i,j,k) << std::endl;
    }}}
}

#endif
