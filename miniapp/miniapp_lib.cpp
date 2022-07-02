#include <cstdio>
#include <cstdlib>

#include "miniapp.hpp"
#include "miniapp_lib.hpp"

//! ----------------------------------------------------------------------------
//! the main miniapp function that is exposed to the shared lib
//! ----------------------------------------------------------------------------
extern "C" void miniapp_lib(const std::string& device_name,
                            const std::string& eos_name,
                            const std::string& model_path,
                            int stop_cycle, bool pack_sparse_mats,
                            int num_mats, int num_elems, int num_qpts,
                            TypeValue* density_in, TypeValue* energy_in,
                            bool* indicators_in) {

    MiniApp<TypeValue> miniapp(num_mats, num_elems, num_qpts, device_name, pack_sparse_mats);
    miniapp.setup(eos_name, model_path);
    miniapp.evaluate(stop_cycle, density_in, energy_in, indicators_in);
}

//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
