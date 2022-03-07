#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mfem.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "eos.hpp"
#include "eos_idealgas.hpp"
#include "miniapp.hpp"

double unitrand() { return (double)rand() / RAND_MAX; }

//! ----------------------------------------------------------------------------
//! entry point to the mini app
//! ----------------------------------------------------------------------------
int main(int argc, char **argv) {

    // create an object of the miniapp
    MiniApp miniapp(argc, argv);

    // set up a randomization seed
    srand(miniapp.seed);

    // setup device
    mfem::Device device(miniapp.device_name);
    device.Print();
    printf("\n");

    // setup eos
    for (int k = 0; k < miniapp.num_mats; ++k)
    {
        miniapp.eoses[k] = new IdealGas(1.6, 1.4);
    }

    // setup indicators
    mfem::Array<bool> indicators_arr(miniapp.num_elems * miniapp.num_mats);
    indicators_arr = false;

    // only needed on host
    auto indicators = mfem::Reshape(indicators_arr.HostReadWrite(), miniapp.num_elems, miniapp.num_mats);
    for (int k = 0; k < miniapp.num_mats; ++k)
    {
        // min ratio if empty_element_ratio is -1
        const double min_ratio = .2;
        const double ratio     = miniapp.empty_element_ratio == -1 ? unitrand() * (1 - min_ratio) + min_ratio
                                                                    : 1 - miniapp.empty_element_ratio;
        const int num_nonzero_elems = ratio * miniapp.num_elems;
        printf("using %d/%d elements for material %d\n", num_nonzero_elems, miniapp.num_elems, k);

        for (int i = 0, nz = 0; i < miniapp.num_elems; ++i)
        {
            if (nz < num_nonzero_elems)
            {
                if (((num_nonzero_elems - nz) == (miniapp.num_elems - i)) || unitrand() <= ratio)
                {
                    indicators(i, k) = true;
                    nz++;
                }
            }
        }
    }

    // inputs
    mfem::DenseTensor density(miniapp.num_qpts, miniapp.num_elems, miniapp.num_mats);
    mfem::DenseTensor energy(miniapp.num_qpts, miniapp.num_elems, miniapp.num_mats);

    // outputs
    mfem::DenseTensor pressure(miniapp.num_qpts, miniapp.num_elems, miniapp.num_mats);
    mfem::DenseTensor soundspeed2(miniapp.num_qpts, miniapp.num_elems, miniapp.num_mats);
    mfem::DenseTensor bulkmod(miniapp.num_qpts, miniapp.num_elems, miniapp.num_mats);
    mfem::DenseTensor temperature(miniapp.num_qpts, miniapp.num_elems, miniapp.num_mats);

    density = 0;
    energy = 0;

    // init inputs
    // TODO: what are good initial values?
    for (int k = 0; k < miniapp.num_mats; ++k)
    {
        for (int i = 0; i < miniapp.num_elems; ++i)
        {
            if (!indicators(i, k))
            {
                continue;
            }
            for (int j = 0; j < miniapp.num_qpts; ++j)
            {
                density(j, i, k) = .1 + unitrand();
                energy(j, i, k)  = .1 + unitrand();
            }
        }
    }

    //miniapp.print_tensor(pressure, indicators, "pressure");

    // run through the cycles (time-steps)
    printf("\n");
    for (int c = 0; c <= miniapp.stop_cycle; ++c)
    {
        printf("cycle %d\n", c);
        miniapp.evaluate(density, energy, indicators,
                         pressure, soundspeed2, bulkmod, temperature);
    }


    //miniapp.print_tensor(pressure, indicators, "pressure");
    return 0;
}

//! ----------------------------------------------------------------------------
