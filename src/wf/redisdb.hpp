#ifndef __AMS_REDIS_DB__
#define __AMS_REDIS_DB__

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <chrono>

#include "basedb.hpp"

#include <sw/redis++/redis++.h>
using namespace sw::redis;

#include "resource_manager.hpp"

template <typename Tin, typename Tout>
class RedisDB : public BaseDB<Tin, Tout> {
    const std::string _fn;         // path to the file storing the DB access config
    uint64_t _dbid;
    Redis* _redis;
public:

    // constructor
    RedisDB(std::string fn) : _fn(fn), _redis(nullptr) {
      _dbid = reinterpret_cast<uint64_t>(this);
      auto connection_info = read_json(fn);

      ConnectionOptions connection_options;
      connection_options.type = ConnectionType::TCP;
      connection_options.host = connection_info["host"];
      connection_options.port = std::stoi(connection_info["service-port"]);
      connection_options.password = connection_info["database-password"];
      connection_options.db = 0; //Optionnal, 0 is the default
      connection_options.tls.enabled = true; // Required to connect to PDS within LC
      connection_options.tls.cacert = connection_info["cert"];

      ConnectionPoolOptions pool_options;
      pool_options.size = 100; // Pool size, i.e. max number of connections.
  
      _redis = new Redis(connection_options, pool_options);   
    }

    ~RedisDB() {
      std::cerr << "Deleting RedisDB object\n";
      delete _redis;
    }
    
    inline std::string type() {
      return "RedisDB";
    }

    inline std::string info() {
      return _redis->info();
    }
    
    // Return the number of keys in the DB
    inline long long dbsize() {
      return _redis->dbsize();
    }

    /* !
     * ! WARNING: Flush the entire Redis, accross all DBs!
     * !
     */
    inline void flushall() {
      _redis->flushall();
    }

    /*
     * ! WARNING: Flush the entire current DB!
     * !
     */
    inline void flushdb() {
      _redis->flushdb();
    }

    std::unordered_map<std::string, std::string> read_json(std::string fn) {
      std::ifstream config;
      std::unordered_map<std::string, std::string> connection_info = {
        {"database-password", ""},
        {"host", ""},
        {"service-port", ""},
        {"cert", ""},
      };

      config.open(fn, std::ifstream::in);
      if (config.is_open()) {
        std::string line;
        // Quite inefficient parsing (to say the least..) but the file to parse is small (4 lines)
        // TODO: maybe use Boost or another JSON library 
        while (std::getline(config, line)) {
          if (line.find("{") != std::string::npos || line.find("}") != std::string::npos) { continue; }
          line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
          line.erase(std::remove(line.begin(), line.end(), ','), line.end());
          line.erase(std::remove(line.begin(), line.end(), '"'), line.end());
  
          std::string key = line.substr(0, line.find(':'));
          line.erase(0, line.find(":") + 1);
          connection_info[key] = line;
          //std::cerr << "key=" << key << " and value=" << line << std::endl;
        }
        config.close();
      } 
      else {
        std::cerr << "Config located at: " << fn << std::endl;
        throw std::runtime_error("Could not open Redis config file");
      }
      return connection_info;
    }

    void store(int iter, size_t num_elements,
               std::vector<Tin*> inputs,
               std::vector<Tout*> outputs) {

      const size_t num_in = inputs.size();
      const size_t num_out = outputs.size();

      // TODO: 
      //      Make insertion more efficient. 
      //      Right now it's pretty naive and expensive 
      auto start = std::chrono::high_resolution_clock::now();
      
      for (size_t i = 0; i < num_elements; i++) {
        std::string key = std::to_string(_dbid)+":"+std::to_string(iter)+":"+std::to_string(i); // In Redis a key must be a string 
        std::ostringstream fd;
        for (size_t j = 0; j < num_in; j++) {
          fd << inputs[j][i] << ":";
        }
        for (size_t j = 0; j < num_out - 1; j++) {
          fd << outputs[j][i] << ":";
        }
        fd << outputs[num_out - 1][i];
        std::string val(fd.str());
        
        
        _redis->set(key, val);
      }

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      auto nb_keys = this->dbsize();

      std::cout << std::setprecision(2) << "Inserted " << num_elements 
        << " keys [Total keys = " << nb_keys << "]  into RedisDB [Total " << duration.count() << "ms, " 
        << static_cast<double>(num_elements)/duration.count() << " key/ms]" << std::endl;
    }
};

#endif // __AMS_REDIS_DB__
