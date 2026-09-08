#include "wf/basedb.hpp"

#include <memory>
#include <string>

#include "AMS.h"
#include "wf/jsondb.hpp"

namespace ams
{
namespace db
{
/**
   * @brief get the type of the database from a string
   * @param[in] type the type of the db in string format
   * @retrun the database type
   */
AMSDBType getDBType(std::string type)
{
  if (type.compare("hdf5") == 0) {
    return AMSDBType::AMS_HDF5;
  } else if (type.compare("rmq") == 0) {
    return AMSDBType::AMS_RMQ;
  } else if (type.compare("json") == 0) {
    return AMSDBType::AMS_JSON;
  }
  return AMSDBType::AMS_NONE;
}

std::string getDBTypeAsStr(AMSDBType type)
{
  switch (type) {
    case AMSDBType::AMS_NONE:
      return "None";
    case AMSDBType::AMS_HDF5:
      return "hdf5";
    case AMSDBType::AMS_RMQ:
      return "rmq";
    case AMSDBType::AMS_JSON:
      return "json";
  }
  return "Unknown";
}

std::shared_ptr<BaseDB> DBManager::createDB(std::string& domainName,
                                            AMSDBType dbType,
                                            uint64_t rId)
{
  AMS_DBG(DBManager, "Instantiating data base");

  if ((dbType == AMSDBType::AMS_HDF5 || dbType == AMSDBType::AMS_JSON) &&
      !fs_interface.isConnected()) {
    THROW(std::runtime_error,
          "File System is not configured, Please specify output directory");
  } else if (dbType == AMSDBType::AMS_RMQ && !rmq_interface.isConnected()) {
    THROW(std::runtime_error, "Rabbit MQ data base is not configured");
  }

  switch (dbType) {
#ifdef __AMS_ENABLE_HDF5__
    case AMSDBType::AMS_HDF5:
      return std::make_shared<hdf5DB>(fs_interface.path(), domainName, rId);
#endif
#ifdef __AMS_ENABLE_RMQ__
    case AMSDBType::AMS_RMQ:
      return std::make_shared<RabbitMQDB>(rmq_interface,
                                          domainName,
                                          rId,
                                          updateSurrogate);
#endif
    case AMSDBType::AMS_JSON: {
      // JSONDB needs json_mode configuration - get from environment or default
      const char* json_mode_env = std::getenv("AMS_JSON_MODE");
      std::string json_mode = json_mode_env ? json_mode_env : "binary";
      return std::make_shared<JSONDB>(fs_interface.path(),
                                      domainName,
                                      rId,
                                      json_mode);
    }
    default:
      return nullptr;
  }
  return nullptr;
}

}  // namespace db
}  // namespace ams
