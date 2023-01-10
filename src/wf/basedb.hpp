#ifndef __AMS_BASE_DB__
#define __AMS_BASE_DB__

#include <vector>
#include <string>

template <typename Tin, typename Tout>
class BaseDB {

public:
    /* Define the type of the DB (File, Redis etc) */
    virtual std::string type() = 0;
    
    /*
    We need to decide how data are represented.
    As an array of structures AOS or as a Structure of
    arrays. Currently the mini app has support of the latter.
    */
    virtual void store(int iter, size_t num_elements,
               std::vector<Tin*> inputs,
               std::vector<Tout*> outputs) = 0;
};

#endif  // __AMS_BASE_DB__
