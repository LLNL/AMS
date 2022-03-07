#ifndef _EOS_HPP_
#define _EOS_HPP_


//! Abstract EOS class
//! Code given by Thomas Stitt
class EOS {

public:
   virtual void Eval(const int length,
                     const double *density,
                     const double *energy,
                     double *pressure,
                     double *soundspeed2,
                     double *bulkmod,
                     double *temperature) = 0;
};

#endif
