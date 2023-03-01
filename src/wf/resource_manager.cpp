// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include "resource_manager.hpp"

namespace ams {

  //! --------------------------------------------------------------------------
  const char*
  ResourceManager::getDeviceAllocatorName() {  return "mmp-device-quickpool"; }

  const char*
  ResourceManager::getHostAllocatorName() {    return "mmp-host-quickpool";   }

  const char*
  ResourceManager::getPinnedAllocatorName() {    return "mmp-pinned-quickpool";   }

  const char *
  ResourceManager::getAllocatorName(AMSResourceType Resource){
    if ( Resource == AMSResourceType::HOST )
      return ResourceManager::getHostAllocatorName();
    else if ( Resource == AMSResourceType::DEVICE )
      return ResourceManager::getDeviceAllocatorName();
    else if ( Resource == AMSResourceType::PINNED)
      return ResourceManager::getPinnedAllocatorName();
    else{
      throw std::runtime_error("Request allocator for resource that does not exist\n");
      return nullptr;
    }
  }

  //! --------------------------------------------------------------------------
  // maintain a list of allocator ids
  int ResourceManager::allocator_ids[AMSResourceType::RSEND] = {-1, -1, -1};

  // maintain a list of allocator names
  std::string ResourceManager::allocator_names[AMSResourceType::RSEND] = { "HOST", "DEVICE", "PINNED" };

  // default allocator
  AMSResourceType ResourceManager::default_resource = AMSResourceType::HOST;

  //! --------------------------------------------------------------------------
  void
  ResourceManager::setDefaultDataAllocator(AMSResourceType location) {
      ResourceManager::default_resource = location;

      auto& rm = umpire::ResourceManager::getInstance();
      auto alloc = rm.getAllocator(allocator_ids[location]);

      std::cout << "  > Setting default allocator: " << alloc.getId() << " : " << alloc.getName() << "\n";
      rm.setDefaultAllocator(alloc);
  }

  AMSResourceType
  ResourceManager::getDefaultDataAllocator() {
      return ResourceManager::default_resource;
  }

  bool
  ResourceManager::isDeviceExecution() {
    return ResourceManager::default_resource == AMSResourceType::DEVICE;
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
ResourceManager::setup(const AMSResourceType Resource) {
    if ( Resource < AMSResourceType::HOST || Resource >= AMSResourceType::RSEND ){
      throw std::runtime_error("Resource does not exist\n");
    }

    std::cout << "\nSetting up ams::ResourceManager("<<allocator_names[Resource]<<")\n";

    // use umpire resource manager
    auto& rm = umpire::ResourceManager::getInstance();

    // -------------------------------------------------------------------------
    // create host allocator
    auto alloc_name = ResourceManager::getAllocatorName(Resource);
    auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>
                                    (alloc_name, rm.getAllocator(allocator_names[Resource]));

    std::cout << "  > Created allocator[" << Resource << "] = "
              << alloc_resource.getId() << ": " << alloc_resource.getName() << "\n";

    allocator_ids[Resource] = alloc_resource.getId();
  }
}  // namespace AMS
