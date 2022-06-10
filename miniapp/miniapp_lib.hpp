#ifndef __MMP_HPP__
#define __MMP_HPP__

//! ----------------------------------------------------------------------------
//! miniapp library interface that can be called from python as well as main.cpp
//! ----------------------------------------------------------------------------

extern "C" void miniapp_lib(const std::string &device_name,
                            const std::string &eos_name,
                            const std::string &model_path,
                            int stop_cycle, bool pack_sparse_mats,
                            int num_mats, int num_elems, int num_qpts,
                            double *density_in, double *energy_in, bool *indicators_in);

#endif
