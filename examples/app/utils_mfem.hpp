/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_UTILS_MFEM_HPP__
#define __AMS_UTILS_MFEM_HPP__

#include <iostream>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"

//! -----------------------------------------------------------------------
#define mfemReshapeTensor3(m, op) \
  mfem::Reshape(m.op(), m.SizeI(), m.SizeJ(), m.SizeK())
#define mfemReshapeArray3(m, op, X, Y, Z) mfem::Reshape(m.op(), X, Y, Z);
#define mfemReshapeArray2(m, op, X, Y) mfem::Reshape(m.op(), X, Y);
#define mfemReshapeArray1(m, op) mfem::Reshape(m.op(), m.Size());

//! -----------------------------------------------------------------------

template <typename T>
using dt1 = mfem::DeviceTensor<1, T>;
template <typename T>
using dt2 = mfem::DeviceTensor<2, T>;
template <typename T>
using dt3 = mfem::DeviceTensor<3, T>;


//! -----------------------------------------------------------------------
//! packing code for mfem tensors (i,j,k)
//! -----------------------------------------------------------------------
//! for us, index j is sparse wrt k
//!     i.e., for a given k, certain j are inactive
//! so we pack a sparse (i,j,k) tensor into a dense (i,j*) tensor
//!     for a given k
//!     where j* is the linearized "dense" index of all sparse indices "j"


template <typename Tin, typename Tout>
static inline void pack_ij(const int k,
                           const int sz_i,
                           const int sz_sparse_j,
                           const int offset_sparse_j,
                           const int *sparse_j_indices,
                           const dt3<Tin> &a3,
                           const dt2<Tout> &a2,
                           const dt3<Tin> &b3,
                           const dt2<Tout> &b2)
{
  using mfem::ForallWrap;
  MFEM_FORALL(j, sz_sparse_j, {
    const int sparse_j = sparse_j_indices[offset_sparse_j + j];
    for (int i = 0; i < sz_i; ++i) {
      a2(i, j) = a3(i, sparse_j, k);
      b2(i, j) = b3(i, sparse_j, k);
    }
  });
}

template <typename Tin, typename Tout>
static inline void unpack_ij(const int k,
                             const int sz_i,
                             const int sz_sparse_j,
                             const int offset_sparse_j,
                             const int *sparse_j_indices,
                             const dt2<Tin> &a2,
                             const dt3<Tout> &a3,
                             const dt2<Tin> &b2,
                             const dt3<Tout> &b3,
                             const dt2<Tin> &c2,
                             const dt3<Tout> &c3,
                             const dt2<Tin> &d2,
                             const dt3<Tout> &d3)
{

  using mfem::ForallWrap;
  MFEM_FORALL(j, sz_sparse_j, {
    const int sparse_j = sparse_j_indices[offset_sparse_j + j];
    for (int i = 0; i < sz_i; ++i) {
      a3(i, sparse_j, k) = a2(i, j);
      b3(i, sparse_j, k) = b2(i, j);
      c3(i, sparse_j, k) = c2(i, j);
      d3(i, sparse_j, k) = d2(i, j);
    }
  });
}


//! ----------------------------------------------------------------------------
//! printing utilitizes for mfem datastructures
//! ----------------------------------------------------------------------------

template <typename T>
static void print_array(const std::string &label, const mfem::Array<T> &values)
{

  std::cout << "--> printing [sz = " << values.Size() << "] array \"" << label
            << "\" = " << values << "\n";
  for (int i = 0; i < values.Size(); ++i) {
    std::cout << label << "[" << i << "] = " << values[i] << std::endl;
  }
}

// -----------------------------------------------------------------------------

void print_dense_tensor(const std::string &label,
                        const mfem::DenseTensor &values)
{

  const int I = values.SizeI(), J = values.SizeJ(), K = values.SizeK();
  std::cerr << "--> printing [" << I << " x " << J << " x " << K
            << "] dense_tensor \"" << label << "\""
            << std::endl;  // = " << values << "\n";

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
      for (int i = 0; i < I; ++i) {
        std::cerr << label << "[" << i << "," << j << ", " << k
                  << "] = " << values(i, j, k) << std::endl;
      }
    }
  }
}

void print_dense_tensor(const std::string &label,
                        const mfem::DenseTensor &values,
                        const mfem::DeviceTensor<2, bool> &filter)
{

  const int I = values.SizeI(), J = values.SizeJ(), K = values.SizeK();
  std::cout << "--> printing [" << I << " x " << J << " x " << K
            << "] dense_tensor \"" << label
            << "\"\n";  // = " << values << "\n";

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
      if (!filter(j, k)) {
        continue;
      }
      for (int i = 0; i < I; ++i) {
        std::cout << label << "[" << i << "," << j << ", " << k
                  << "] = " << values(i, j, k) << "\n";
      }
    }
  }
}

void print_dense_tensor(const std::string &label,
                        const mfem::DenseTensor &values,
                        const bool *filter)
{

  const int I = values.SizeI(), J = values.SizeJ(), K = values.SizeK();
  std::cout << "--> printing [" << I << " x " << J << " x " << K
            << "] dense_tensor \"" << label
            << "\"\n";  // = " << values << "\n";

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < J; ++j) {
      if (!filter[j + k * J]) continue;
      for (int i = 0; i < I; ++i) {
        std::cout << label << "[" << i << "," << j << ", " << k
                  << "] = " << values(i, j, k) << "\n";
      }
    }
  }
}

// -----------------------------------------------------------------------------
template <typename T>
static void print_device_tensor(const std::string &label,
                                const mfem::DeviceTensor<2, T> &values,
                                const std::array<int, 2> &sz)
{

  const int I = sz[0], J = sz[1];
  std::cout << "--> printing [" << I << " x " << J << "] device_tensor \""
            << label << "\" = " << values << "\n";
  for (int i = 0; i < sz[0]; ++i) {
    for (int j = 0; j < sz[1]; ++j) {
      std::cout << label << "[" << i << "][" << j << "] = " << values(i, j)
                << std::endl;
    }
  }
}

template <typename T>
static void print_device_tensor(const std::string &label,
                                const mfem::DeviceTensor<3, T> &values,
                                const std::array<int, 3> &sz)
{

  const int I = sz[0], J = sz[1], K = sz[2];
  std::cout << "--> printing [" << I << " x " << J << " x " << K
            << "] device_tensor \"" << label << "\"\n";
  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
      for (int k = 0; k < K; ++k) {
        std::cout << label << "[" << i << "][" << j << "][" << k
                  << "] = " << values(i, j, k) << std::endl;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
static void print_tensor_array(const std::string &label,
                               const T *values,
                               const std::array<int, 3> &sz)
{

  const int K = sz[0], J = sz[1], I = sz[2];
  if (K == 1) {
    std::cout << "--> printing [" << J << " x " << I << "] tensor \"" << label
              << "\"\n";
    for (int j = 0; j < J; ++j) {
      for (int i = 0; i < I; ++i) {
        int idx = i + I * j;
        std::cout << label << "[" << j << "," << i << "] = " << idx << " = "
                  << values[idx] << std::endl;
      }
    }
  } else {
    std::cout << "--> printing [" << K << " x " << J << " x " << I
              << "] tensor \"" << label << "\"\n";
    for (int k = 0; k < K; ++k) {
      for (int j = 0; j < J; ++j) {
        for (int i = 0; i < I; ++i) {
          int idx = i + I * (j + k * J);
          std::cout << label << "[" << k << ", " << j << "," << i
                    << "] = " << values[idx] << std::endl;
        }
      }
    }
  }
}


//!
#endif
