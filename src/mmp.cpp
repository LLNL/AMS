#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "eos.hpp"
#include "eos_idealgas.hpp"
#include "surrogate.hpp"
#include "hdcache.hpp"
#include "miniapp.hpp"
#include "mfem_utils.hpp"


//! ----------------------------------------------------------------------------
//! the main miniapp function that is exposed to the shared lib
extern "C"
void mmp_main(bool is_cpu, const char *device_name, int stop_cycle, bool pack_sparse_mats,
             int num_mats, int num_elems, int num_qpts, const char *model_path,
             double *density_in, double *energy_in, bool *indicators_in) {


    // create an object of the miniapp
    MiniApp miniapp(num_mats, num_elems, num_qpts, is_cpu, pack_sparse_mats);

    // setup device

    mfem::Device device(device_name);
    //mfem::Device device("cpu");
    device.Print();
    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // setup eos, surrogare models, and hdcaches
    // TODO: keeping it consistent with Tom's existing code currently
    // Do we want different surrogates and caches for different materials?
    IdealGas *temp_eos = new IdealGas(1.6, 1.4);
    int cache_dim = 2;
    for (int mat_idx = 0; mat_idx < miniapp.num_mats; ++mat_idx) {
        miniapp.eoses[mat_idx] = new IdealGas(1.6, 1.4);
        miniapp.surrogates[mat_idx] = new SurrogateModel(temp_eos, model_path, is_cpu);
        miniapp.hdcaches[mat_idx] = new HDCache(cache_dim);
    }

    // -------------------------------------------------------------------------
    // initialize inputs and outputs as mfem tensors
    // -------------------------------------------------------------------------
    // inputs

    // mfem::DenseTensor has shapw (i,j,k)
    //  contains k 2D "DenseMatrix" each of size ixj

#if 0
    // if the data has been defined in C order (mat x elems x qpts)
    // the following conversion works
    mfem::DenseTensor density(density_in, num_qpts, num_elems, num_mats);
    mfem::DenseTensor energy(energy_in, num_qpts, num_elems, num_mats);

#else
    mfem::DenseTensor density(num_qpts, num_elems, num_mats);
    mfem::DenseTensor energy(num_mats, num_elems, num_qpts);

    // init inputs
    density = 0;
    energy = 0;

    // TODO: what are good initial values?
    int idx = 0;
    for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {

        const int me = mat_idx*num_elems + elem_idx;
        if (!indicators_in[me]) {
            continue;
        }
        for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx) {
            density(qpt_idx, elem_idx, mat_idx) = density_in[qpt_idx + me*num_qpts];
            energy(qpt_idx, elem_idx, mat_idx)  = energy_in[qpt_idx + me*num_qpts];
        }
    }}
#endif

    if (0) {
        //print_tensor_array("indicators_in", indicators_in, {1, num_mats, num_elems});
        //print_tensor_array("density_in", density_in, {num_mats, num_elems, num_qpts});
        //print_tensor_array("energy_in", energy_in, {num_mats, num_elems, num_qpts});

        //print_dense_tensor("density", density);
        //print_dense_tensor("density", density, indicators_in);
        //print_dense_tensor("energy", energy);
        //print_dense_tensor("energy", energy, indicators_in);
    }

    // outputs
    mfem::DenseTensor pressure(num_qpts, num_elems, num_mats);
    mfem::DenseTensor soundspeed2(num_qpts, num_elems, num_mats);
    mfem::DenseTensor bulkmod(num_qpts, num_elems, num_mats);
    mfem::DenseTensor temperature(num_qpts, num_elems, num_mats);

    // -------------------------------------------------------------------------
    // run through the cycles (time-steps)
    printf("\n");
    miniapp.start();
    for (int c = 0; c <= stop_cycle; ++c)
    {
        std::cout << "--> cycle " << c << std::endl;
        miniapp.evaluate(density, energy, indicators_in,
                         pressure, soundspeed2, bulkmod, temperature);
        //break;
    }

    //print_dense_tensor("pressure", pressure, indicators_in);
}


//! ----------------------------------------------------------------------------
//! ----------------------------------------------------------------------------
