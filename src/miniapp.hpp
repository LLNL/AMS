#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "eos.hpp"
#include "surrogate.hpp"
#include "hdcache.hpp"

#include "mfem_utils.hpp"


//! ----------------------------------------------------------------------------
//! mini app class
//! ----------------------------------------------------------------------------
class MiniApp {

public:
    bool is_cpu = true;
    bool pack_sparse_mats = true;
    int num_mats = 1;
    int num_elems = 1;
    int num_qpts = 1;

    std::vector<EOS *> eoses;

    // added to include ML
    std::vector<HDCache *> hdcaches;
    std::vector<SurrogateModel *> surrogates;

    // -------------------------------------------------------------------------
    // constructor and destructor
    // -------------------------------------------------------------------------
    MiniApp(int _num_mats, int _num_elems, int _num_qpts,
            bool _is_cpu, bool _pack_sparse_mats) {

        is_cpu = _is_cpu;
        num_mats = _num_mats;
        num_elems = _num_elems;
        num_qpts = _num_qpts;
        pack_sparse_mats = _pack_sparse_mats;

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
                  //mfem::DeviceTensor<2, bool>& indicators,
                  const bool *indicators,
                  mfem::DenseTensor &pressure,
                  mfem::DenseTensor &soundspeed2,
                  mfem::DenseTensor &bulkmod,
                  mfem::DenseTensor &temperature) {


        // move/allocate data on the device.
        // if the data is already on the device this is basically a noop
        const auto d_density     = RESHAPE_TENSOR(density, Read);
        const auto d_energy      = RESHAPE_TENSOR(energy, Read);
        const auto d_pressure    = RESHAPE_TENSOR(pressure, Write);
        const auto d_soundspeed2 = RESHAPE_TENSOR(soundspeed2, Write);
        const auto d_bulkmod     = RESHAPE_TENSOR(bulkmod, Write);
        const auto d_temperature = RESHAPE_TENSOR(temperature, Write);

        for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {

            // TODO: this is repeated computation in each cycle
            // cant think of any reason we should do this
            // we perhaps dont even need indicators on the device
            int num_elems_for_mat = 0;
            for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
                //num_elems_for_mat += indicators(elem_idx, mat_idx);
                num_elems_for_mat += indicators[elem_idx + num_elems*mat_idx];
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
                    //if (indicators(elem_idx, mat_idx)) {
                     if (indicators[elem_idx + num_elems*mat_idx]) {
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
                    for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx) {
                        d_dense_density(qpt_idx, elem_idx) = d_density(qpt_idx, sparse_elem_idx, mat_idx);
                        d_dense_energy(qpt_idx, elem_idx)  = d_energy(qpt_idx, sparse_elem_idx, mat_idx);
                    }
                });

                // TODO: I think Tom mentiond we can allocate these outside the loop
                // check again
                mfem::Array<double> dense_pressure(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_soundspeed2(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_bulkmod(num_elems_for_mat * num_qpts);
                mfem::Array<double> dense_temperature(num_elems_for_mat * num_qpts);

                auto d_dense_pressure = mfem::Reshape(dense_pressure.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_soundspeed2 = mfem::Reshape(dense_soundspeed2.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_bulkmod = mfem::Reshape(dense_bulkmod.Write(), num_qpts, num_elems_for_mat);
                auto d_dense_temperature = mfem::Reshape(dense_temperature.Write(), num_qpts, num_elems_for_mat);

                // create uq flags
                // ask Tom about the memory management for this
                // should we create this memory again and again?
                mfem::Array<bool> dense_uq(num_elems_for_mat * num_qpts);
                dense_uq = false;
                auto d_dense_uq = mfem::Reshape(dense_uq.Write(), num_qpts, num_elems_for_mat);


                // STEP 1:
                // call the hdcache to look at input uncertainties
                // to decide if making a ML inference makes sense
                hdcaches[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                                                &d_dense_density(0, 0),
                                                &d_dense_energy(0, 0),
                                                &d_dense_uq(0, 0));


                // STEP 2:
                // let's call surrogate for everything
                surrogates[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                                          &d_dense_density(0, 0),
                                          &d_dense_energy(0, 0),
                                          &d_dense_pressure(0, 0),
                                          &d_dense_soundspeed2(0, 0),
                                          &d_dense_bulkmod(0, 0),
                                          &d_dense_temperature(0, 0));


                // STEP 3:
                // let's call physics module only where the d_dense_uq = flags are true
                eoses[mat_idx]->Eval_with_filter(num_elems_for_mat * num_qpts,
                                                 &d_dense_density(0, 0),
                                                 &d_dense_energy(0, 0),
                                                 &d_dense_uq(0, 0),
                                                 &d_dense_pressure(0, 0),
                                                 &d_dense_soundspeed2(0, 0),
                                                 &d_dense_bulkmod(0, 0),
                                                 &d_dense_temperature(0, 0));


                // STEP 3: convert dense -> sparse
                MFEM_FORALL(elem_idx, num_elems_for_mat, {
                   const int sparse_elem_idx = d_sparse_index[elem_idx];
                   for (int qpt_idx = 0; qpt_idx < num_qpts; ++qpt_idx) {

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
};

