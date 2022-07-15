#include "allocator.hpp"
#include <umpire/Umpire.hpp>

namespace ams {

  //! --------------------------------------------------------------------------
  const std::string
  ResourceManager::getDeviceAllocatorName() {  return "mmp-device-quickpool"; }

  const std::string
  ResourceManager::getHostAllocatorName() {    return "mmp-host-quickpool";   }

  //! --------------------------------------------------------------------------
  // maintain a list of allocator ids
  int ResourceManager::allocator_ids[ResourceType::RSEND] = {-1, -1};

  // default allocator
  ResourceManager::ResourceType ResourceManager::default_resource = ResourceManager::ResourceType::HOST;

  //! --------------------------------------------------------------------------
  void
  ResourceManager::setDefaultDataAllocator(ResourceManager::ResourceType location) {
      ResourceManager::default_resource = location;

      auto& rm = umpire::ResourceManager::getInstance();
      auto alloc = rm.getAllocator(allocator_ids[location]);

      std::cout << "  > Setting default allocator: " << alloc.getId() << " : " << alloc.getName() << "\n";
      rm.setDefaultAllocator(alloc);
  }

  ResourceManager::ResourceType
  ResourceManager::getDefaultDataAllocator() {
      return ResourceManager::default_resource;
  }

  bool
  ResourceManager::isDeviceExecution() {
    return ResourceManager::default_resource == ResourceManager::ResourceType::DEVICE;
  }


// -----------------------------------------------------------------------------
// get the list of available allocators
// -----------------------------------------------------------------------------
void
ResourceManager::list_allocators() {

  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc_names = rm.getAllocatorNames();
  auto alloc_ids = rm.getAllocatorIds();

  std::cout << "  > Listing data allocators registered with ams::ResourceManager\n";
  for (int i = 0; i < std::max(alloc_ids.size(), alloc_names.size()); i++) {

    if (i < alloc_ids.size() && i < alloc_names.size()) {
      std::cout << "     [id = "<<alloc_ids[i]<<"] name = " << alloc_names[i]<<"\n";
    }
    else if (i < alloc_names.size()) {  // id not available
      std::cout << "     [id = ?] name = "<<alloc_names[i]<<"\n";
    }
    else {                              // name not available
      std::cout << "     [id = "<<alloc_ids[i]<<"] name = ?\n";
    }
  }

  auto dalloc = rm.getDefaultAllocator();
  std::cout << "  > Default allocator = (" << dalloc.getId() << " : " << dalloc.getName() << ")\n";
}


// -----------------------------------------------------------------------------
// set up the resource manager
// -----------------------------------------------------------------------------
void
ResourceManager::setup(const std::string &device_name) {

    std::cout << "\nSetting up ams::ResourceManager("<<device_name<<")\n";
    const bool use_device = device_name != "cpu";

    // use umpire resource manager
    auto& rm = umpire::ResourceManager::getInstance();

    // -------------------------------------------------------------------------
    // create host allocator
    auto alloc_name_host = ResourceManager::getHostAllocatorName();
    auto alloc_host = rm.makeAllocator<umpire::strategy::QuickPool, true>
                                    (alloc_name_host, rm.getAllocator("HOST"));

    std::cout << "  > Created allocator[" << ResourceType::HOST << "] = "
              << alloc_host.getId() << ": " << alloc_host.getName() << "\n";

    allocator_ids[ResourceType::HOST] = alloc_host.getId();
    setDefaultDataAllocator(ResourceType::HOST);

    if (use_device) {
        auto alloc_name_device = ResourceManager::getDeviceAllocatorName();
        auto alloc_device = rm.makeAllocator<umpire::strategy::QuickPool, true>
                                          (alloc_name_device, rm.getAllocator("DEVICE"));

        std::cout << "  > Created allocator[" << ResourceType::DEVICE << "] = "
                  << alloc_device.getId() << ": " << alloc_device.getName() << "\n";

        allocator_ids[ResourceType::DEVICE] = alloc_device.getId();
        setDefaultDataAllocator(ResourceType::DEVICE);
    }
  }
}  // namespace AMS
