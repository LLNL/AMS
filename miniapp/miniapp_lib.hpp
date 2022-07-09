#ifndef __MINIAPP_LIB_HPP__
#define __MINIAPP_LIB_HPP__

#include <string>

using TypeValue = double;

//! ----------------------------------------------------------------------------
//! miniapp library interface that can be called from python as well as main.cpp
//! ----------------------------------------------------------------------------

extern "C" void miniapp_lib(const std::string &device_name,
                            const std::string &eos_name,
                            const std::string &model_path,
                            const std::string &hdcache_path,
                            int stop_cycle, bool pack_sparse_mats,
                            int num_mats, int num_elems, int num_qpts,
                            TypeValue *density_in, TypeValue *energy_in,
                            bool *indicators_in);

//! ----------------------------------------------------------------------------
#endif
