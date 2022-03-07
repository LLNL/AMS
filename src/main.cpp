#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/dtensor.hpp"

#include "eos.hpp"
#include "eos_idealgas.hpp"

#define RESHAPE_TENSOR(m, op) mfem::Reshape(m.op(), m.SizeI(), m.SizeJ(), m.SizeK())


double unitrand() { return (double)rand() / RAND_MAX; }

int main(int argc, char **argv)
{
   const char *device_name    = "cpu";
   int stop_cycle             = 10;
   int num_mats               = 5;
   int num_elems              = 10000;
   int num_qpts               = 64;
   double empty_element_ratio = -1;
   int seed                   = 0;
   bool pack_sparse_mats      = true;

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
      return 1;
   }
   args.PrintOptions(std::cout);

   // small validation
   assert(stop_cycle > 0);
   assert(num_mats > 0);
   assert(num_elems > 0);
   assert(num_qpts > 0);
   assert((empty_element_ratio >= 0 && empty_element_ratio < 1) || empty_element_ratio == -1);

   srand(seed);

   // setup device
   mfem::Device device(device_name);
   device.Print();
   printf("\n");
   bool is_cpu = std::string(device_name) == "cpu";

   // setup eos
   std::vector<EOS *> eoses(num_mats);
   for (int k = 0; k < num_mats; ++k)
   {
      eoses[k] = new IdealGas(1.6, 1.4);
   }

   // setup indicators
   mfem::Array<bool> indicators_arr(num_elems * num_mats);
   indicators_arr = false;
   // only needed on host
   auto indicators = mfem::Reshape(indicators_arr.HostReadWrite(), num_elems, num_mats);
   for (int k = 0; k < num_mats; ++k)
   {
      // min ratio if empty_element_ratio is -1
      const double min_ratio = .2;
      const double ratio     = empty_element_ratio == -1 ? unitrand() * (1 - min_ratio) + min_ratio
                                                     : 1 - empty_element_ratio;
      const int num_nonzero_elems = ratio * num_elems;
      printf("using %d/%d elements for material %d\n", num_nonzero_elems, num_elems, k);
      for (int i = 0, nz = 0; i < num_elems; ++i)
      {
         if (nz < num_nonzero_elems)
         {
            if (((num_nonzero_elems - nz) == (num_elems - i)) || unitrand() <= ratio)
            {
               indicators(i, k) = true;
               nz++;
            }
         }
      }
   }

   // inputs
   mfem::DenseTensor density(num_qpts, num_elems, num_mats);
   density = 0;
   mfem::DenseTensor energy(num_qpts, num_elems, num_mats);
   energy = 0;

   // init inputs
   // TODO: what are good initial values?
   for (int k = 0; k < num_mats; ++k)
   {
      for (int i = 0; i < num_elems; ++i)
      {
         if (!indicators(i, k))
         {
            continue;
         }
         for (int j = 0; j < num_qpts; ++j)
         {
            density(j, i, k) = .1 + unitrand();
            energy(j, i, k)  = .1 + unitrand();
         }
      }
   }

   // outputs
   mfem::DenseTensor pressure(num_qpts, num_elems, num_mats);
   mfem::DenseTensor soundspeed2(num_qpts, num_elems, num_mats);
   mfem::DenseTensor bulkmod(num_qpts, num_elems, num_mats);
   mfem::DenseTensor temperature(num_qpts, num_elems, num_mats);

   for (int c = 0; c <= stop_cycle; ++c)
   {
      printf("cycle %d\n", c);

      // move/allocate data on the device. if the data is already on the device this is basically a noop
      const auto d_density     = RESHAPE_TENSOR(density, Read);
      const auto d_energy      = RESHAPE_TENSOR(energy, Read);
      const auto d_pressure    = RESHAPE_TENSOR(pressure, Write);
      const auto d_soundspeed2 = RESHAPE_TENSOR(soundspeed2, Write);
      const auto d_bulkmod     = RESHAPE_TENSOR(bulkmod, Write);
      const auto d_temperature = RESHAPE_TENSOR(temperature, Write);

      for (int k = 0; k < num_mats; ++k)
      {
         int num_elems_k = 0;
         for (int i = 0; i < num_elems; ++i)
         {
            num_elems_k += indicators(i, k);
         }

         if (num_elems_k == 0)
         {
            continue;
         }
         // NOTE: we've found it's faster to do sparse lookups on GPUs but on CPUs the dense
         // packing->looked->unpacking is better if we're using expensive eoses. in the future
         // we may just use dense representations everywhere but for now we use sparse ones.
         else if (is_cpu && pack_sparse_mats && num_elems_k < num_elems)
         {
            using mfem::ForallWrap;

            mfem::Array<int> sparse_index(num_elems_k);
            for (int i = 0, nz = 0; i < num_elems; ++i)
            {
               if (indicators(i, k))
               {
                  sparse_index[nz++] = i;
               }
            }
            const auto *d_sparse_index = sparse_index.Read();

            mfem::Array<double> dense_density(num_elems_k * num_qpts);
            auto d_dense_density = mfem::Reshape(dense_density.Write(), num_qpts, num_elems_k);
            mfem::Array<double> dense_energy(num_elems_k * num_qpts);
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
            auto d_dense_pressure = mfem::Reshape(dense_pressure.Write(), num_qpts, num_elems_k);
            mfem::Array<double> dense_soundspeed2(num_elems_k * num_qpts);
            auto d_dense_soundspeed2 = mfem::Reshape(dense_soundspeed2.Write(),
                                                     num_qpts,
                                                     num_elems_k);
            mfem::Array<double> dense_bulkmod(num_elems_k * num_qpts);
            auto d_dense_bulkmod = mfem::Reshape(dense_bulkmod.Write(), num_qpts, num_elems_k);
            mfem::Array<double> dense_temperature(num_elems_k * num_qpts);
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
         else
         {
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

   for (int k = 0; k < num_mats; ++k)
   {
      delete eoses[k];
   }

   return 0;
}
