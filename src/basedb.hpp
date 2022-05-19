#ifndef __BASE_DB__
#define __BASE_DB__
#include <fstream>
#include <iostream>

// A simple class writting data
// into a file. At some point we should
// convert this into a pure virtual class.
#ifdef __ENABLE_DB__
class BaseDB {
public:
  // path to the file storing the db data
  string fn;
  // "Emulated data base through a file"
  fstream fd;

  BaseDB(string fn) : fn(fn) {
    fd.open(fn, std::ios_base::app | std::ios_base::in);
    if (!fd.is_open()) {
      std::cerr << "Cannot open db file\n";
    }
  }
  ~BaseDB() {}

  /*
  We need to decide how data are represented.
  As an array of structures AOS or as a Structure of
  arrays. Currently the mini app has support of the latter.
  */
  template <class ITYPE, class OTYPE>
  void Store(size_t num_elements, size_t num_in, size_t num_out, ITYPE **inputs,
             OTYPE **outputs) {
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
#else
    class BaseDB {
    public:
      // path to the file storing the db data
      std::string fn;
      // "Emulated data base through a file"
      std::fstream fd;
      //
      BaseDB(string fn) : fn(fn) {
        // void
      }
      ~BaseDB() {}

      template <class ITYPE, class OTYPE>
      void Store(size_t num_elements, size_t num_in, size_t num_out,
                 ITYPE **inputs, OTYPE **outputs) {}
    };

#endif

#endif