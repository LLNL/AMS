#ifndef _HDCACHE_HPP_
#define _HDCACHE_HPP_

#include <cstdint>
#include <cstdlib>

//! An implementation of hdcache
//! the idea is to have a single class that exposes
//! the functionality to compute uncertainty
class HDCache {

    using idx_dist_pair = std::pair<int, double>;

    //! dimensionality of the space
    uint8_t dim;

    //! a private low-level function to compute the nearest neighbors
    //! currently stored in the hdcache
    // TODO: currently, using a single input vector
    // but, really, this should be a matrix if size (N, D)
    // N query points, and D features
    std::vector<idx_dist_pair>
    find_nearest_neighbors(const size_t length, const double *input) const {

       // return a vector of tuples: (index of, distance to) the nearest nbr
       std::vector<idx_dist_pair> nnbrs (length);
        for (size_t i = 0; i < length; i++) {
            nnbrs[i].first = -1;
            nnbrs[i].second = ((double)rand() / RAND_MAX);
        }
        return nnbrs;
    }


public:
   HDCache(uint8_t _dim) {
       // TODO: create an empty faiss index
       dim = _dim;
   }

   //! -----------------------------------------------------------------------
   void load_cache(const std::string &filename) {
        // TODO: load a pre-trained and possibly pre-populated index
   }
   void save_cache(const std::string &filename) {
        // TODO: save the current index
   }

   //! -----------------------------------------------------------------------
   bool add_to_cache(const int length, const double *input) {
       // TODO: should append these points to the cache
       // so we can use them in the next query
   }

   //! -----------------------------------------------------------------------
   //! this function works on the "inputs"
   //! to see how good are our error estimates in the input space
   //! i.e., we should call this *before* ML inference
   void Eval(const int length,
             const double *density,
             const double *energy,
             bool *is_acceptable)  const {

       // should pass all features for hdcache
       auto nnbrs = find_nearest_neighbors(length, density);

       const double acceptable_error = 0.5;
       for(int i = 0; i < length; i++) {
           is_acceptable[i] = nnbrs[i].second <= acceptable_error;
           //std::cout << i << " " << nnbrs[i].second << " " << is_acceptable[i] << "\n";
       }
   }


   //! this function can use both "inputs" and "outputs"
   //! to estimate uncertainity in either or both
   //! i.e., we should call this *after* ML inference
   void Eval(const int length,
             const double *density,
             const double *energy,
             double *pressure,
             double *soundspeed2,
             double *bulkmod,
             bool *is_acceptable)  const {

       // should pass all features for hdcache
       auto nnbrs = find_nearest_neighbors(length, density);

       const double acceptable_error = 0.5;
       for(int i = 0; i < length; i++) {
           is_acceptable[i] = nnbrs[i].second <= acceptable_error;
       }
   }

    //! -----------------------------------------------------------------------
};

#endif
