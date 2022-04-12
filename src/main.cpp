#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <mfem.hpp>
#include <mfem/linalg/dtensor.hpp>

#include "eos.hpp"
#include "eos_idealgas.hpp"
#include "surrogate.hpp"
#include "hdcache.hpp"
#include "miniapp.hpp"
#include "mfem_utils.hpp"

double unitrand() { return (double)rand() / RAND_MAX; }


//! ----------------------------------------------------------------------------
//! the main miniapp function that is exposed to the shared lib
extern "C"
int mmp_main(bool is_cpu, int stop_cycle, bool pack_sparse_mats,
             int num_mats, int num_elems, int num_qpts,
             double *density_in, double *energy_in, bool *indicators_in) {


    // create an object of the miniapp
    MiniApp miniapp(num_mats, num_elems, num_qpts, is_cpu, pack_sparse_mats);

    // setup device

    //mfem::Device device(miniapp.device_name);
    mfem::Device device("cpu");
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
        miniapp.surrogates[mat_idx] = new SurrogateModel(temp_eos);
        miniapp.hdcaches[mat_idx] = new HDCache(cache_dim);
    }

    // -------------------------------------------------------------------------
    // initialize inputs and outputs as mfem tensors
    // -------------------------------------------------------------------------
    // inputs

    // mfem::DenseTensor has shapw (i,j,k)
    //  contains k 2D "DenseMatrix" each of size ixj

#if 1
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

        print_dense_tensor("density", density);
        print_dense_tensor("density", density, indicators_in);
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
struct MiniAppArgs {

    const char *device_name    = "cpu";
    bool is_cpu                = true;

    int seed                   = 0;
    double empty_element_ratio = -1;

    int stop_cycle             = 10;

    int num_mats               = 5;
    int num_elems              = 10000;
    int num_qpts               = 64;
    bool pack_sparse_mats      = true;

    bool parse(int argc, char** argv) {

        mfem::OptionsParser args(argc, argv);
        args.AddOption(&device_name, "-d", "--device", "Device config string");
        args.AddOption(&stop_cycle, "-c", "--stop-cycle", "Stop cycle");
        args.AddOption(&num_mats, "-m", "--num-mats", "Number of materials");
        args.AddOption(&num_elems, "-e", "--num-elems", "Number of elements");
        args.AddOption(&num_qpts, "-q", "--num-qpts", "Number of quadrature points per element");
        args.AddOption(&empty_element_ratio,
                       "-r",
                       "--empty-element-ratio",
                       "Fraction of elements that are empty "
                       "for each material. If -1 use a random value for each. ");
        args.AddOption(&seed, "-s", "--seed", "Seed for rand");
        args.AddOption(&pack_sparse_mats,
                       "-p",
                       "--pack-sparse",
                       "-np",
                       "--do-not-pack-sparse",
                       "pack sparse material data before evals (cpu only)");

        args.Parse();
        if (!args.Good())
        {
           args.PrintUsage(std::cout);
           return false;
        }
        args.PrintOptions(std::cout);

        // small validation
        assert(stop_cycle > 0);
        assert(num_mats > 0);
        assert(num_elems > 0);
        assert(num_qpts > 0);
        assert((empty_element_ratio >= 0 && empty_element_ratio < 1) || empty_element_ratio == -1);

        return true;
    }
};


//! ----------------------------------------------------------------------------
//! entry point to the mini app
//! ----------------------------------------------------------------------------
int main(int argc, char **argv) {

    // order of indicators is [Mat x Elem]
    // order of all data is   [Mat x Elem x Qpt]

    MiniAppArgs args;
    bool success = args.parse(argc, argv);
    if (!success) {
        exit(EXIT_FAILURE);
    }

    // set up a randomization seed
    srand(args.seed);

    // -------------------------------------------------------------------------
    // setup indicators
    //  to represent which combinations of materials and elements exist
    // -------------------------------------------------------------------------
    bool indicators[args.num_mats * args.num_elems];

    for (int mat_idx = 0; mat_idx < args.num_mats; ++mat_idx) {

        // min ratio if empty_element_ratio is -1
        const double min_ratio = 0.2;
        const double ratio     = args.empty_element_ratio == -1 ? unitrand() * (1 - min_ratio) + min_ratio
                                                                 : 1 - args.empty_element_ratio;
        const int num_nonzero_elems = ratio * args.num_elems;
        std::cout << "using " << num_nonzero_elems << "/"<< args.num_elems << " for material " << mat_idx << std::endl;

        int nz = 0;
        for (int elem_idx = 0; elem_idx < args.num_elems; ++elem_idx) {

            const int me = elem_idx + mat_idx*args.num_elems;
            indicators[me] = false;

            if (nz < num_nonzero_elems) {
                if (((num_nonzero_elems - nz) == (args.num_elems - elem_idx)) || unitrand() <= ratio) {
                    indicators[me] = true;
                    std::cout << " setting (mat = " << mat_idx << ", elem = " << elem_idx << ") = 1\n";
                    nz++;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // initialize inputs
    // -------------------------------------------------------------------------
    // inputs
    std::vector<double> density (args.num_mats * args.num_elems * args.num_qpts, 0.);
    std::vector<double> energy (args.num_mats * args.num_elems * args.num_qpts, 0.);

    // init inputs
    // TODO: what are good initial values?
    for (int mat_idx = 0; mat_idx < args.num_mats; ++mat_idx) {
        for (int elem_idx = 0; elem_idx < args.num_elems; ++elem_idx) {

           const int me = mat_idx*args.num_elems + elem_idx;
            if (!indicators[me])
                continue;

            for (int qpt_idx = 0; qpt_idx < args.num_qpts; ++qpt_idx) {
                //density[qpt_idx + me*args.num_qpts] = 100*mat_idx + 10*elem_idx + qpt_idx;
                //energy[qpt_idx + me*args.num_qpts] = 100*mat_idx + 10*elem_idx + qpt_idx;
                density[qpt_idx + me*args.num_qpts] = .1 + unitrand();
                energy[qpt_idx + me*args.num_qpts] = .1 + unitrand();
            }
        }
    }

    // -------------------------------------------------------------------------
    mmp_main(args.is_cpu, args.stop_cycle, args.pack_sparse_mats,
             args.num_mats, args.num_elems, args.num_qpts,
             density.data(), energy.data(), indicators);


    return EXIT_SUCCESS;
}

//! ----------------------------------------------------------------------------
