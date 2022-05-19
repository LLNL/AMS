#ifndef __MMP_HPP__
#define __MMP_HPP__

//! ----------------------------------------------------------------------------
//! miniapp library interface that can be called from python as well as main.cpp
//! ----------------------------------------------------------------------------

extern "C"
void mmp_main(bool is_cpu, const char *device_name, int stop_cycle, bool pack_sparse_mats,
             int num_mats, int num_elems, int num_qpts, const char *model_path,
             double *density_in, double *energy_in, bool *indicators_in);

#endif
