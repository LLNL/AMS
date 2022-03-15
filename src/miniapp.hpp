#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "eos.hpp"
#include "surrogate.hpp"
#include "hdcache.hpp"

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

    // added to include ML
    std::vector<HDCache *> hdcaches;
    std::vector<SurrogateModel *> surrogates;

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
        hdcaches.resize(num_mats, nullptr);
        surrogates.resize(num_mats, nullptr);
    }

    ~MiniApp() {
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            delete eoses[mat_idx];
            delete hdcaches[mat_idx];
            delete surrogates[mat_idx];
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

        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {

            int num_elems_for_mat = 0;
            for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
                num_elems_for_mat += indicators(elem_idx, mat_idx);
            }

            if (num_elems_for_mat == 0) {
                continue;
            }

            // NOTE: we've found it's faster to do sparse lookups on GPUs but on CPUs the dense
            // packing->looked->unpacking is better if we're using expensive eoses. in the future
            // we may just use dense representations everywhere but for now we use sparse ones.
            else if (is_cpu && pack_sparse_mats && num_elems_for_mat < num_elems) {

                printf(" material %d: using sparse packing for %lu elems\n", mat_idx, num_elems_for_mat);

                // compute sparse indices
                using mfem::ForallWrap;
                mfem::Array<int> sparse_index(num_elems_for_mat);
                for (int elem_idx = 0, nz = 0; elem_idx < num_elems; ++elem_idx) {
                    if (indicators(elem_idx, mat_idx)) {
                        sparse_index[nz++] = elem_idx;
                    }
                }

                const auto *d_sparse_index = sparse_index.Read();

                mfem::Array<double> dense_density(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_energy(num_elems_for_mat * num_qpts);

                auto d_dense_density = mfem::Reshape(dense_density.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_energy = mfem::Reshape(dense_energy.Write(), num_qpts, num_elems_for_mat);

                // sparse -> dense
                MFEM_FORALL(elem_idx, num_elems_for_mat, {
                    const int sparse_elem_idx = d_sparse_index[elem_idx];
                    for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx)
                    {
                        d_dense_density(qpt_idx, elem_idx) = d_density(qpt_idx, sparse_elem_idx, mat_idx);
                        d_dense_energy(qpt_idx, elem_idx)  = d_energy(qpt_idx, sparse_elem_idx, mat_idx);
                    }
                });

                mfem::Array<double> dense_pressure(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_soundspeed2(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_bulkmod(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_temperature(num_elems_for_mat * num_qpts);

                auto d_dense_pressure = mfem::Reshape(dense_pressure.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_soundspeed2 = mfem::Reshape(dense_soundspeed2.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_bulkmod = mfem::Reshape(dense_bulkmod.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_temperature = mfem::Reshape(dense_temperature.Write(), num_qpts, num_elems_for_mat);


                // create for uq flags
                // ask Tom about the memory management for this
                // should we create this memory again and again?
                mfem::Array<bool> dense_uq(num_elems_for_mat * num_qpts);
                auto d_dense_uq = mfem::Reshape(dense_uq.Write(), num_qpts, num_elems_for_mat);


                // STEP 1:
                // call the hdcache to look at input uncertainties
                // to decide if making a ML inference makes sense
                hdcaches[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                                                &d_dense_density(0, 0),
                                                &d_dense_energy(0, 0),
                                                &d_dense_uq(0, 0));

                // STEP 2:
                // slide the data based on d_dense_uq flag
                // TODO: need help from Tom


                // STEP 3a:
                // for d_dense_uq = False, we call surrogate
                surrogates[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                                          &d_dense_density(0, 0),
                                          &d_dense_energy(0, 0),
                                          &d_dense_pressure(0, 0),
                                          &d_dense_soundspeed2(0, 0),
                                          &d_dense_bulkmod(0, 0),
                                          &d_dense_temperature(0, 0));


                // STEP 3b:
                // for d_dense_uq = True, we call physics
                eoses[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                                     &d_dense_density(0, 0),
                                     &d_dense_energy(0, 0),
                                     &d_dense_pressure(0, 0),
                                     &d_dense_soundspeed2(0, 0),
                                     &d_dense_bulkmod(0, 0),
                                     &d_dense_temperature(0, 0));

                // dense -> sparse
                MFEM_FORALL(elem_idx, num_elems_for_mat, {
                   const int sparse_elem_idx = d_sparse_index[elem_idx];
                   for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx)
                   {
                      d_pressure(qpt_idx, sparse_elem_idx, mat_idx)    = d_dense_pressure(qpt_idx, elem_idx);
                      d_soundspeed2(qpt_idx, sparse_elem_idx, mat_idx) = d_dense_soundspeed2(qpt_idx, elem_idx);
                      d_bulkmod(qpt_idx, sparse_elem_idx, mat_idx)     = d_dense_bulkmod(qpt_idx, elem_idx);
                      d_temperature(qpt_idx, sparse_elem_idx, mat_idx) = d_dense_temperature(qpt_idx, elem_idx);
                   }
                });
         }

            else {
                printf(" material %d: using dense packing for %lu elems\n", mat_idx, num_elems_for_mat);
                eoses[mat_idx]->Eval(num_elems * num_qpts,
                                     &d_density(0, 0, mat_idx),
                                     &d_energy(0, 0, mat_idx),
                                     &d_pressure(0, 0, mat_idx),
                                     &d_soundspeed2(0, 0, mat_idx),
                                     &d_bulkmod(0, 0, mat_idx),
                                     &d_temperature(0, 0, mat_idx));
            }
        }
    }


    void print_tensor(const mfem::DenseTensor &values,
                      const mfem::DeviceTensor<2, bool>& indicators,
                      const std::string &label) {

        printf("-- printing tensor (%s)\n", label.c_str());
        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
            for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
                if (indicators(elem_idx, mat_idx)) {
                    for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx) {
                        printf("%s[%d][%d][%d] = %f\n", label.c_str(),
                               qpt_idx, elem_idx, mat_idx, values(qpt_idx,elem_idx,mat_idx));
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

