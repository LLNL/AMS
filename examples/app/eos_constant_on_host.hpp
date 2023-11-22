/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef EOS_CONSTANT_ON_HOST_HPP
#define EOS_CONSTANT_ON_HOST_HPP

#include <iostream>
#include <umpire/Umpire.hpp>

#include "eos.hpp"

template <typename FPType>
class ConstantEOSOnHost : public EOS<FPType>
{
  umpire::ResourceManager &rm_;
  const char *host_allocator_name_;
  const FPType val_;

  static const char *GetResourceName(camp::resources::Platform platform)
  {
    using camp::resources::Platform;
    switch (platform) {
      case Platform::cuda:
        return "cuda";
      case Platform::hip:
        return "hip";
      case Platform::host:
        return "host";
      default:
        return "unknown";
    }
  }

  camp::resources::Platform GetPlatform(void *ptr) const
  {
    // TODO: just let Umpire raise an exception instead of this check?
    if (!rm_.hasAllocator(ptr)) {
      std::cerr << "ConstantEOSOnHost::GetPlatform: Pointer not known by Umpire"
                << std::endl;
      exit(1);
    }
    const auto *record = rm_.findAllocationRecord(ptr);
    return record->strategy->getPlatform();
  }

public:
  ConstantEOSOnHost(const char *host_allocator_name, FPType val)
      : rm_(umpire::ResourceManager::getInstance()),
        host_allocator_name_(host_allocator_name),
        val_(val)
  {
  }

#ifdef __ENABLE_PERFFLOWASPECT__
  __attribute__((annotate("@critical_path(pointcut='around')")))
#endif
  void
  Eval(const int length,
       const FPType *density,
       const FPType *energy,
       FPType *pressure,
       FPType *soundspeed2,
       FPType *bulkmod,
       FPType *temperature) const override
  {
    Eval_with_filter(length,
                     density,
                     energy,
                     nullptr,
                     pressure,
                     soundspeed2,
                     bulkmod,
                     temperature);
  }

#ifdef __ENABLE_PERFFLOWASPECT__
  __attribute__((annotate("@critical_path(pointcut='around')")))
#endif
  void
  Eval_with_filter(const int length,
                   const FPType *density,
                   const FPType *energy,
                   const bool *filter,
                   FPType *pressure,
                   FPType *soundspeed2,
                   FPType *bulkmod,
                   FPType *temperature) const override
  {
    auto plt = GetPlatform((void *)density);
    const char *res_name = GetResourceName(plt);

    FPType *h_density, *h_energy;
    bool *h_filter = nullptr;
    FPType *h_pressure, *h_soundspeed2, *h_bulkmod, *h_temperature;

    // NOTE: probably better to check if cuda or hip or sycl since omp is the
    // host mem space
    if (plt != camp::resources::Platform::host) {
      std::cerr << "Memory is in the " << res_name
                << " memory space, moving to host" << std::endl;

      auto allocator = rm_.getAllocator(host_allocator_name_);

      // not needed for constant eos but hey let's do it anyways
      h_density = (FPType *)allocator.allocate(length * sizeof(FPType));
      h_energy = (FPType *)allocator.allocate(length * sizeof(FPType));
      rm_.copy(h_density,
               const_cast<FPType *>(density),
               length * sizeof(FPType));
      rm_.copy(h_energy, const_cast<FPType *>(energy), length * sizeof(FPType));

      if (filter) {
        h_filter = (bool *)allocator.allocate(length * sizeof(bool));
        rm_.copy(h_filter, const_cast<bool *>(filter), length * sizeof(bool));
      }

      h_pressure = (FPType *)allocator.allocate(length * sizeof(FPType));
      h_soundspeed2 = (FPType *)allocator.allocate(length * sizeof(FPType));
      h_bulkmod = (FPType *)allocator.allocate(length * sizeof(FPType));
      h_temperature = (FPType *)allocator.allocate(length * sizeof(FPType));

    } else {
      std::cerr << "Memory is on the host, nothing special to do" << std::endl;

      h_density = const_cast<FPType *>(density);
      h_energy = const_cast<FPType *>(energy);

      h_filter = const_cast<bool *>(filter);

      h_pressure = pressure;
      h_soundspeed2 = soundspeed2;
      h_bulkmod = bulkmod;
      h_temperature = temperature;
    }

    for (int i = 0; i < length; ++i) {
      // Unused
      // const FPType density = h_density[i];
      // const FPType energy = h_energy[i];
      if (filter && h_filter[i]) {
        continue;
      }
      h_pressure[i] = val_;
      h_soundspeed2[i] = val_;
      h_bulkmod[i] = val_;
      h_temperature[i] = val_;
    }

    if (plt != camp::resources::Platform::host) {
      std::cerr << "Moving back to " << res_name << std::endl;

      rm_.copy(pressure, h_pressure, length * sizeof(FPType));
      rm_.copy(soundspeed2, h_soundspeed2, length * sizeof(FPType));
      rm_.copy(bulkmod, h_bulkmod, length * sizeof(FPType));
      rm_.copy(temperature, h_temperature, length * sizeof(FPType));

      rm_.deallocate(h_temperature);
      rm_.deallocate(h_bulkmod);
      rm_.deallocate(h_soundspeed2);
      rm_.deallocate(h_pressure);
      if (filter) {
        rm_.deallocate(h_filter);
      }
      rm_.deallocate(h_energy);
      rm_.deallocate(h_density);
    }
  }
};

#endif  // EOS_CONSTANT_ON_HOST_HPP
