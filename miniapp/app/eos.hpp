#ifndef _EOS_HPP_
#define _EOS_HPP_


//! Abstract EOS class
//! Code given by Thomas Stitt
class EOS {

public:
   virtual void
   Eval(const int length,
        const double *density,
        const double *energy,
        double *pressure,
        double *soundspeed2,
        double *bulkmod,
        double *temperature) const = 0;


    virtual void
    Eval_with_filter(const int length,
                     const double *density,
                     const double *energy,
                     const bool *filter,
                     double *pressure,
                     double *soundspeed2,
                     double *bulkmod,
                     double *temperature) const = 0;

   virtual void computeRMSE(const int length,
        const double *density,
        const double *energy,
        double *pressure,
        double *soundspeed2,
        double *bulkmod,
        double *temperature) const = 0;
};

#endif
