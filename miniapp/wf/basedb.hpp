#ifndef __AMS_BASE_DB__
#define __AMS_BASE_DB__

#include <string>
#include <vector>
#include <fstream>
#include <iostream>


//! ----------------------------------------------------------------------------
//! A simple class writing data into a file.
//! At some point we should convert this into a pure virtual class.
//! ----------------------------------------------------------------------------

class BaseDB {

public:
    const std::string fn;         // path to the file storing the db data
    std::fstream fd;              // "Emulated data base through a file"

    // constructor
    BaseDB(std::string fn) : fn(fn) {

#ifdef __ENABLE_DB__
        fd.open(fn, std::ios_base::app | std::ios_base::in);
        if (!fd.is_open()) {
            std::cerr << "Cannot open db file\n";
        }
#endif
    }
    ~BaseDB() {}

    /*
    We need to decide how data are represented.
    As an array of structures AOS or as a Structure of
    arrays. Currently the mini app has support of the latter.
    */
    template <typename Tin, typename Tout>
    void Store(size_t num_elements,
               std::vector<Tin*> inputs,
               std::vector<Tout*> outputs) {

#ifdef __ENABLE_DB__
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
#endif
    }
};

#endif
