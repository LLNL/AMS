#include <string>

#include "AMS.h"

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
    return AMSDBType::HDF5;
  } else if (type.compare("csv") == 0) {
    return AMSDBType::CSV;
  } else if (type.compare("redis") == 0) {
    return AMSDBType::REDIS;
  } else if (type.compare("rmq") == 0) {
    return AMSDBType::RMQ;
  }
  return AMSDBType::None;
}


}  // namespace db
}  // namespace ams
