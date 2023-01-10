#ifndef __AMS_FILE_DB__
#define __AMS_FILE_DB__

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "basedb.hpp"

template <typename Tin, typename Tout>
class FileDB : public BaseDB<Tin, Tout> {

public:
    const std::string fn;         // path to the file storing the db data
    std::fstream fd;              // "Emulated data base through a file"

    // constructor
    FileDB(std::string fn) : fn(fn) {
        fd.open(fn, std::ios_base::app | std::ios_base::out);
        if (!fd.is_open()) {
            std::cerr << "Cannot open db file: " << fn << std::endl;
        }
    }
    
    ~FileDB() {
        std::cerr << "Deleting FileDB object\n";
        fd.close();
    }

    std::string type() {
      return "FileDB";
    }

    void store(int iter, size_t num_elements,
               std::vector<Tin*> inputs,
               std::vector<Tout*> outputs) {

        const size_t num_in = inputs.size();
        const size_t num_out = outputs.size();

        for (size_t i = 0; i < num_elements; i++) {
            for (size_t j = 0; j < num_in; j++) {
                fd << inputs[j][i] << ":";
            }
            for (size_t j = 0; j < num_out - 1; j++) {
                fd << outputs[j][i] << ":";
            }
            fd << outputs[num_out - 1][i] << "\n";
        }
    }
};

#endif // __AMS_FILE_DB__
