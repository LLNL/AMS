#ifndef EOS_CONSTANT_ON_HOST_HPP
#define EOS_CONSTANT_ON_HOST_HPP

#include <iostream>
#include <umpire/Umpire.hpp>
#include "eos.hpp"

class ConstantEOSOnHost : public EOS {
    umpire::ResourceManager &rm_;
    const char *host_allocator_name_;
    const double val_;

    static const char *GetResourceName(camp::resources::Platform platform) {
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

    camp::resources::Platform GetPlatform(void *ptr) const {
        // TODO: just let Umpire raise an exception instead of this check?
        if (!rm_.hasAllocator(ptr)) {
            std::cerr << "ConstantEOSOnHost::GetPlatform: Pointer not known by Umpire" << std::endl;
            exit(1);
        }
        const auto *record = rm_.findAllocationRecord(ptr);
        return record->strategy->getPlatform();
    }

  public:
    ConstantEOSOnHost(const char *host_allocator_name, double val)
        : rm_(umpire::ResourceManager::getInstance()), host_allocator_name_(host_allocator_name),
          val_(val) {}

    void Eval(const int length, const double *density, const double *energy, double *pressure,
              double *soundspeed2, double *bulkmod, double *temperature) const override {
        Eval_with_filter(length, density, energy, nullptr, pressure, soundspeed2, bulkmod,
                         temperature);
    }

    void Eval_with_filter(const int length, const double *density, const double *energy,
                          const bool *filter, double *pressure, double *soundspeed2,
                          double *bulkmod, double *temperature) const override {
        auto plt = GetPlatform((void *)density);
        const char *res_name = GetResourceName(plt);

        double *h_density, *h_energy;
        bool *h_filter = nullptr;
        double *h_pressure, *h_soundspeed2, *h_bulkmod, *h_temperature;

        // NOTE: probably better to check if cuda or hip or sycl since omp is the host mem space
        if (plt != camp::resources::Platform::host) {
            std::cerr << "Memory is in the " << res_name << " memory space, moving to host"
                      << std::endl;

            auto allocator = rm_.getAllocator(host_allocator_name_);

            // not needed for constant eos but hey let's do it anyways
            h_density = (double *)allocator.allocate(length * sizeof(double));
            h_energy = (double *)allocator.allocate(length * sizeof(double));
            rm_.copy(h_density, const_cast<double *>(density), length * sizeof(double));
            rm_.copy(h_energy, const_cast<double *>(energy), length * sizeof(double));

            if (filter) {
                h_filter = (bool *)allocator.allocate(length * sizeof(bool));
                rm_.copy(h_filter, const_cast<bool *>(filter), length * sizeof(bool));
            }

            h_pressure = (double *)allocator.allocate(length * sizeof(double));
            h_soundspeed2 = (double *)allocator.allocate(length * sizeof(double));
            h_bulkmod = (double *)allocator.allocate(length * sizeof(double));
            h_temperature = (double *)allocator.allocate(length * sizeof(double));

        } else {
            std::cerr << "Memory is on the host, nothing special to do" << std::endl;

            h_density = const_cast<double *>(density);
            h_energy = const_cast<double *>(energy);

            h_filter = const_cast<bool *>(filter);

            h_pressure = pressure;
            h_soundspeed2 = soundspeed2;
            h_bulkmod = bulkmod;
            h_temperature = temperature;
        }

        for (int i = 0; i < length; ++i) {
            const double density = h_density[i];
            const double energy = h_energy[i];
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

            rm_.copy(pressure, h_pressure, length * sizeof(double));
            rm_.copy(soundspeed2, h_soundspeed2, length * sizeof(double));
            rm_.copy(bulkmod, h_bulkmod, length * sizeof(double));
            rm_.copy(temperature, h_temperature, length * sizeof(double));

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

    void computeRMSE(const int length, const double *density, const double *energy,
                     double *pressure, double *soundspeed2, double *bulkmod,
                     double *temperature) const override {
        std::cerr << "implement" << std::endl;
    }
};

#endif // EOS_CONSTANT_ON_HOST_HPP
