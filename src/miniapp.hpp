#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "eos.hpp"

#define RESHAPE_TENSOR(m, op) mfem::Reshape(m.op(), m.SizeI(), m.SizeJ(), m.SizeK())


//! ----------------------------------------------------------------------------
//! mini app class
//! ----------------------------------------------------------------------------
class MiniApp {

public:
    const char *device_name    = "cpu";
    int stop_cycle             = 10;
    int num_mats               = 5;
    int num_elems              = 10000;
    int num_qpts               = 64;
    double empty_element_ratio = -1;
    int seed                   = 0;
    bool pack_sparse_mats      = true;
    bool is_cpu                = true;

    std::vector<EOS *> eoses;

    // -------------------------------------------------------------------------
    // constructor and destructor
    // -------------------------------------------------------------------------
    MiniApp(int argc, char **argv) {

        // parse command line arguments
        bool success = _parse_args(argc, argv);
        if (!success) {
            exit(1);
        }

        is_cpu = std::string(device_name) == "cpu";

        // setup eos
        eoses.resize(num_mats, nullptr);
    }

    ~MiniApp() {
        for (int k = 0; k < num_mats; ++k) {
            delete eoses[k];
        }
    }


    // -------------------------------------------------------------------------
    // the main loop
    // -------------------------------------------------------------------------
    void evaluate(mfem::DenseTensor &density,
                  mfem::DenseTensor &energy,
                  mfem::DeviceTensor<2, bool>& indicators,
                  mfem::DenseTensor &pressure,
                  mfem::DenseTensor &soundspeed2,
                  mfem::DenseTensor &bulkmod,
                  mfem::DenseTensor &temperature) {


        // move/allocate data on the device. if the data is already on the device this is basically a noop
        const auto d_density     = RESHAPE_TENSOR(density, Read);
        const auto d_energy      = RESHAPE_TENSOR(energy, Read);
        const auto d_pressure    = RESHAPE_TENSOR(pressure, Write);
        const auto d_soundspeed2 = RESHAPE_TENSOR(soundspeed2, Write);
        const auto d_bulkmod     = RESHAPE_TENSOR(bulkmod, Write);
        const auto d_temperature = RESHAPE_TENSOR(temperature, Write);

        for (int k = 0; k < num_mats; ++k) {

            int num_elems_k = 0;
            for (int i = 0; i < num_elems; ++i) {
                num_elems_k += indicators(i, k);
            }

            if (num_elems_k == 0) {
                continue;
            }

            // NOTE: we've found it's faster to do sparse lookups on GPUs but on CPUs the dense
            // packing->looked->unpacking is better if we're using expensive eoses. in the future
            // we may just use dense representations everywhere but for now we use sparse ones.
            else if (is_cpu && pack_sparse_mats && num_elems_k < num_elems) {

                printf(" material %d: using sparse packing for %lu elems\n", k, num_elems_k);

                using mfem::ForallWrap;
                mfem::Array<int> sparse_index(num_elems_k);
                for (int i = 0, nz = 0; i < num_elems; ++i) {
                    if (indicators(i, k)) {
                        sparse_index[nz++] = i;
                    }
                }

                const auto *d_sparse_index = sparse_index.Read();

                mfem::Array<double> dense_density(num_elems_k * num_qpts);
                mfem::Array<double> dense_energy(num_elems_k * num_qpts);

                auto d_dense_density = mfem::Reshape(dense_density.Write(), num_qpts, num_elems_k);
                auto d_dense_energy = mfem::Reshape(dense_energy.Write(), num_qpts, num_elems_k);

                // sparse -> dense
                MFEM_FORALL(i, num_elems_k, {
                    const int sparse_i = d_sparse_index[i];
                    for (int j = 0; j < num_qpts; ++j)
                    {
                        d_dense_density(j, i) = d_density(j, sparse_i, k);
                        d_dense_energy(j, i)  = d_energy(j, sparse_i, k);
                    }
                });

                mfem::Array<double> dense_pressure(num_elems_k * num_qpts);
                mfem::Array<double> dense_soundspeed2(num_elems_k * num_qpts);
                mfem::Array<double> dense_bulkmod(num_elems_k * num_qpts);
                mfem::Array<double> dense_temperature(num_elems_k * num_qpts);

                auto d_dense_pressure = mfem::Reshape(dense_pressure.Write(), num_qpts, num_elems_k);
                auto d_dense_soundspeed2 = mfem::Reshape(dense_soundspeed2.Write(),
                                                         num_qpts,
                                                         num_elems_k);
                auto d_dense_bulkmod = mfem::Reshape(dense_bulkmod.Write(), num_qpts, num_elems_k);
                auto d_dense_temperature = mfem::Reshape(dense_temperature.Write(),
                                                         num_qpts,
                                                         num_elems_k);

                eoses[k]->Eval(num_elems_k * num_qpts,
                                       &d_dense_density(0, 0),
                                       &d_dense_energy(0, 0),
                                       &d_dense_pressure(0, 0),
                                       &d_dense_soundspeed2(0, 0),
                                       &d_dense_bulkmod(0, 0),
                                       &d_dense_temperature(0, 0));

                // dense -> sparse
                MFEM_FORALL(i, num_elems_k, {
                   const int sparse_i = d_sparse_index[i];
                   for (int j = 0; j < num_qpts; ++j)
                   {
                      d_pressure(j, sparse_i, k)    = d_dense_pressure(j, i);
                      d_soundspeed2(j, sparse_i, k) = d_dense_soundspeed2(j, i);
                      d_bulkmod(j, sparse_i, k)     = d_dense_bulkmod(j, i);
                      d_temperature(j, sparse_i, k) = d_dense_temperature(j, i);
                   }
                });
         }

            else {
                printf(" material %d: using dense packing for %lu elems\n", k, num_elems_k);
                eoses[k]->Eval(num_elems * num_qpts,
                               &d_density(0, 0, k),
                               &d_energy(0, 0, k),
                               &d_pressure(0, 0, k),
                               &d_soundspeed2(0, 0, k),
                               &d_bulkmod(0, 0, k),
                               &d_temperature(0, 0, k));
            }
        }
    }


    void print_tensor(const mfem::DenseTensor &values,
                      const mfem::DeviceTensor<2, bool>& indicators,
                      const std::string &label) {

        printf("-- printing tensor (%s)\n", label.c_str());
        for (int m = 0; m < num_mats; ++m) {
            for (int e = 0; e < num_elems; ++e) {
                if (indicators(e, m)) {
                    for (int q = 0; q < num_qpts; ++q) {
                        printf("%s[%d][%d][%d] = %f\n", label.c_str(), q, e, m, values(q,e,m));
                    }
                }
            }
        }
    }

private:

    bool _parse_args(int argc, char** argv) {

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

