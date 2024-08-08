#ifndef __AMS_VALIDATOR_HPP__
#define __AMS_VALIDATOR_HPP__
#include <iostream>
#include <string>
#include <cstring> // memcpy
#include <vector>
#include <array>
#include <algorithm>
#include <mpi.h>
#include <type_traits> // is_same

#if __STANDALONE_TEST__
namespace ams
{
using AMSResourceType = int;
struct ResourceManager
{
  template <typename T>
  static T* allocate(size_t n, AMSResourceType appDataLoc) {
    return (T*) malloc (sizeof(T)*n);
  }
  static void deallocate(void* ptr, AMSResourceType appDataLoc) {
    free(ptr);
  }
};
}
#else // for off-line testing
#include "wf/resource_manager.hpp"
#endif

namespace ams
{
template <typename FPTypeValue>
struct ValPoint
{
public:
  ValPoint () : m_pos(0ul), m_o_sur(0.0), m_o_phy(0.0) {}
  ValPoint (size_t pos, FPTypeValue o_sur)
    : m_pos(pos), m_o_sur(o_sur), m_o_phy(0.0) {}
  ValPoint (size_t pos, FPTypeValue o_sur, FPTypeValue o_phy)
    : m_pos(pos), m_o_sur(o_sur), m_o_phy(o_phy) {}

  std::string to_string() const {
    return "[ " + std::to_string(m_pos) +
           ", " + std::to_string(m_o_sur) +
           ", " + std::to_string(m_o_phy) + " ]";
  }

public:
  size_t m_pos;
  FPTypeValue m_o_sur;
  FPTypeValue m_o_phy;
};

template <typename FPTypeValue>
struct ValStats
{
  ValStats()
  : m_cnt({0u,0u}),
    m_avg({static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)}),
    m_var({static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)})
  {}

  ValStats(const std::array<unsigned int, 2>& cnt,
           const std::array<FPTypeValue, 2>& avg,
           const std::array<FPTypeValue, 2>& var)
  : m_cnt(cnt), m_avg(avg), m_var(var)
  {}

  std::array<unsigned int, 2> m_cnt;
  std::array<FPTypeValue, 2> m_avg;
  std::array<FPTypeValue, 2> m_var;
};

template <typename FPTypeValue>
class StepValPoints {
public:
  StepValPoints() : m_step(0ul) {}
  void set_step (const size_t i) { m_step = i; }
  void inc_step () { m_step ++; }
  size_t get_iter () const { return m_step; }
  void add_pt(const ValPoint<FPTypeValue>& pt) { m_val_pts.emplace_back(pt); }
  void add_pt(ValPoint<FPTypeValue>&& pt) { m_val_pts.emplace_back(pt); }

  void sort_points() {
    auto less_pt = [](const ValPoint<FPTypeValue>& a, const ValPoint<FPTypeValue>& b)
    {
      return (a.m_pos < b.m_pos);
    };

    std::sort(m_val_pts.begin(), m_val_pts.end(), less_pt);
  }

  size_t get_num_points() const { return m_val_pts.size(); }

  size_t get_last_pos() const {
    return ((m_val_pts.size() == 0ul)? 0ul : m_val_pts.back().m_pos);
  }

  void print() const {
    std::cout << "step " << m_step << std::endl << "validation points : ";
    for (const auto& pt: m_val_pts) {
      std::cout << ' ' + pt.to_string();
    }
    std::cout << std::endl;
  }
  const std::vector<ValPoint<FPTypeValue>>& val_pts() const { return m_val_pts; }
  std::vector<ValPoint<FPTypeValue>>& val_pts() { return m_val_pts; }

protected:
  size_t m_step;
  std::vector<ValPoint<FPTypeValue>> m_val_pts;
};

/**
 *
 */
template <typename FPTypeValue>
class VPCollector {
public:
  VPCollector(unsigned seed, MPI_Comm comm = MPI_COMM_WORLD);
  ~VPCollector();

  bool set_validation(const bool* pred_loc,
                      size_t num_pred_loc,
                      AMSResourceType appDataLoc,
                      unsigned k_v);
#if __STANDALONE_TEST__
  void backup_surrogate_outs(const FPTypeValue* sur_out,
                             StepValPoints<FPTypeValue>& step_valpoints);
  std::array<FPTypeValue, 2> backup_validation_outs(const FPTypeValue* phy_out,
                                                    StepValPoints<FPTypeValue>& step_valpoints);
  ValStats<FPTypeValue> get_error_stats(const FPTypeValue* phy_out,
                                        StepValPoints<FPTypeValue>& step_valpoints);
#endif
  void backup_surrogate_outs(const std::vector<FPTypeValue*> sur_out,
                             std::vector<StepValPoints<FPTypeValue>>& step_valpoints);

  const bool* predicate() const { return m_pred_loc_new; }

  std::vector<ValStats<FPTypeValue>> get_error_stats(
      const std::vector<FPTypeValue*> phy_out,
      std::vector<StepValPoints<FPTypeValue>>& step_valpoints);

protected:
  int gather_predicate(const bool* pred_loc, size_t num_pred_loc);
  size_t pick_num_val_pts(const size_t n_T, const size_t n_F, unsigned k) const;
  std::vector<size_t> pick_val_pts(unsigned k = 0u);
  void turn_predicate_on(const std::vector<size_t>& val_pts);
  bool* distribute_predicate(const AMSResourceType appDataLoc);
  /// Clear intermediate data for root rank to handle predicate update
  void clear_intermediate_info();
  std::array<FPTypeValue, 2> sum_sqdev(const FPTypeValue* phy_out,
                                       const FPTypeValue* sur_out,
                                       const StepValPoints<FPTypeValue>& step_valpoints,
                                       const std::array<FPTypeValue, 2>& avg);
  std::vector<std::array<FPTypeValue, 2>> backup_validation_outs(
    const std::vector<FPTypeValue*> phy_out,
    std::vector<StepValPoints<FPTypeValue>>& step_valpoints);

protected:
  MPI_Comm m_comm;
  int m_rank;
  int m_num_ranks;
  size_t m_num_pred_loc;
  const bool* m_pred_loc;
  bool* m_pred_loc_new;
  AMSResourceType m_appDataLoc;
  std::vector<FPTypeValue*> m_sur; ///< surrogate output backup

  // For root rank
  std::vector<uint8_t> m_predicate_all;
  std::vector<int> m_num_pred_all;
  std::vector<int> m_rcnts;
  std::vector<int> m_displs;
  std::default_random_engine m_rng;
};

template <typename FPTypeValue>
void VPCollector<FPTypeValue>::clear_intermediate_info()
{
  m_predicate_all.clear();
  m_num_pred_all.clear();
  m_rcnts.clear();
  m_displs.clear();
}

template <typename FPTypeValue>
VPCollector<FPTypeValue>::VPCollector(unsigned seed, MPI_Comm comm)
: m_comm(comm), m_num_pred_loc(0ul), m_pred_loc(nullptr), m_pred_loc_new(nullptr)
{
  MPI_Comm_rank(comm, &m_rank);
  MPI_Comm_size(comm, &m_num_ranks);
  m_rng.seed(seed + 1357u + m_rank);
  m_rng();
}

template <typename FPTypeValue>
VPCollector<FPTypeValue>::~VPCollector()
{
  clear_intermediate_info();
  if (m_pred_loc_new) {
#if __STANDALONE_TEST__
    ams::ResourceManager::deallocate(m_pred_loc_new, m_appDataLoc);
    for (size_t i = 0u; i < m_sur.size(); ++i) {
      ams::ResourceManager::deallocate(m_sur[i], m_appDataLoc);
    }
#else
    auto &rm_d = ams::ResourceManager::getInstance();
    rm_d.deallocate(m_pred_loc_new, m_appDataLoc);
    for (size_t i = 0u; i < m_sur.size(); ++i) {
      rm_d.deallocate(m_sur[i], m_appDataLoc);
    }
#endif // __STANDALONE_TEST__
    m_pred_loc_new = nullptr;
    m_sur.clear();
  }
  m_pred_loc = nullptr;
  m_num_pred_loc = 0ul;
}

/// Gather predicates to root rank
template <typename FPTypeValue>
int VPCollector<FPTypeValue>::gather_predicate(const bool* pred_loc, size_t num_pred_loc)
{
  m_pred_loc = pred_loc;
  m_num_pred_loc = num_pred_loc;
  int cnt_loc = static_cast<int>(num_pred_loc);
  int cnt_all = 0;
  int rc = 0;

  m_num_pred_all.clear();
  m_num_pred_all.resize(m_num_ranks);
  // Gather the data sizes (i.e., the number of items) from each rank
  rc = MPI_Gather(reinterpret_cast<const void*>(&cnt_loc),
                  1,
                  MPI_INT,
                  reinterpret_cast<void*>(m_num_pred_all.data()),
                  1,
                  MPI_INT,
                  0,
                  m_comm);

  if (rc != MPI_SUCCESS) {
    if (m_rank == 0) {
      std::cerr << "MPI_Gather() in gather_predicate() failed with code ("
                << rc << ")" << std::endl;
    }
    return rc;
  }

  m_displs.clear();
  m_rcnts.clear();

  if (m_rank == 0) {
    m_displs.resize(m_num_ranks);
    m_rcnts.resize(m_num_ranks);

    int offset = 0;
    for (int i = 0; i < m_num_ranks; ++i) {
      m_displs[i] = offset;
      offset += (m_rcnts[i] = m_num_pred_all[i]);
    }
    m_predicate_all.resize(cnt_all = offset);
  }

  rc = MPI_Gatherv(reinterpret_cast<const void*>(pred_loc),
                   cnt_loc,
                   MPI_C_BOOL,
                   reinterpret_cast<void*>(m_predicate_all.data()),
                   m_rcnts.data(),
                   m_displs.data(),
                   MPI_UINT8_T,
                   0,
                   m_comm);

  if (rc != MPI_SUCCESS) {
    if (m_rank == 0) {
      std::cerr << "MPI_Gatherv() in gather_predicate() failed with code ("
                << rc << ")" << std::endl;
    }
  }

  return rc;
}

/// Determine the number of points to evaluate physics model on while leveraging workers idle due to load imbalance
template <typename FPTypeValue>
size_t VPCollector<FPTypeValue>::pick_num_val_pts(const size_t n_T, const size_t n_F, unsigned k) const
{
  const size_t imbalance = n_F % m_num_ranks;
  const size_t n_idle = (imbalance == static_cast<size_t>(0ul))? 0ul : (m_num_ranks - imbalance);
  const size_t n_val = std::min (n_idle + k * m_num_ranks, n_T);

  return n_val;
}

/// Randonly choose the points to run physics on out of those accepted with the surrogate
template <typename FPTypeValue>
std::vector<size_t> VPCollector<FPTypeValue>::pick_val_pts(unsigned k)
{
  std::vector<size_t> accepted; // positions of accepted surrogate values
  accepted.reserve(m_predicate_all.size());
  for (size_t i = 0ul; i < m_predicate_all.size(); ++i) {
    if (m_predicate_all[i]) {
      accepted.push_back(i);
    }
  }
  const size_t num_val = pick_num_val_pts(accepted.size(), m_predicate_all.size()-accepted.size(), k);
  std::shuffle(accepted.begin(), accepted.end(), m_rng);
  std::sort(accepted.begin(), accepted.begin() + num_val);

  return std::vector<size_t>(accepted.cbegin(), accepted.cbegin() + num_val);
}

template <typename FPTypeValue>
void VPCollector<FPTypeValue>::turn_predicate_on(const std::vector<size_t>& val_pts)
{
  for (const auto i: val_pts) {
    m_predicate_all[i] = static_cast<uint8_t>(0);
  }
}

template <typename FPTypeValue>
bool* VPCollector<FPTypeValue>::distribute_predicate(
    const AMSResourceType appDataLoc)
{
  int rc = 0;
  m_appDataLoc = appDataLoc;
#if __STANDALONE_TEST__
  m_pred_loc_new = ams::ResourceManager::allocate<bool>(m_num_pred_loc, appDataLoc);
#else
  auto &rm_a = ams::ResourceManager::getInstance();
  m_pred_loc_new = rm_a.allocate<bool>(m_num_pred_loc, appDataLoc);
#endif // __STANDALONE_TEST__

  if (!m_pred_loc_new) {
    if (m_rank == 0) {
      std::cerr << "allocate() in distribute_predicate() failed!" << std::endl;
      clear_intermediate_info();
    }
    return nullptr;
  }

  rc = MPI_Scatterv(
         reinterpret_cast<const void*>(m_predicate_all.data()),
         reinterpret_cast<const int*>(m_rcnts.data()),
         reinterpret_cast<const int*>(m_displs.data()),
         MPI_UINT8_T,
         reinterpret_cast<void*>(m_pred_loc_new),
         static_cast<int>(m_num_pred_loc),
         MPI_C_BOOL,
         0, m_comm);

  if (rc != MPI_SUCCESS) {
    if (m_rank == 0) {
      std::cerr << "MPI_Scatterv() in distribute_predicate() failed with code ("
                << rc << ")" << std::endl;
      clear_intermediate_info();
    }
#if __STANDALONE_TEST__
    ams::ResourceManager::deallocate(m_pred_loc_new, appDataLoc);
#else
    auto &rm_d = ams::ResourceManager::getInstance();
    rm_d.deallocate(m_pred_loc_new, appDataLoc);
#endif // __STANDALONE_TEST__
    return nullptr;
  }

  return m_pred_loc_new;
}

template <typename FPTypeValue>
bool VPCollector<FPTypeValue>::set_validation(
  const bool* pred_loc,
  const size_t num_pred_loc,
  const AMSResourceType appDataLoc,
  const unsigned k_v)
{
  int rc = 0;
  rc = gather_predicate(pred_loc, num_pred_loc);
  if (rc != MPI_SUCCESS) {
    return false;
  }

  if (m_rank == 0) {
    std::vector<size_t> val_pts_all = pick_val_pts(k_v);
   #if 1
    std::cout << "validation points: size(" << val_pts_all.size() << ")";
    for (size_t i = 0ul; i < val_pts_all.size(); i++) {
        std::cout << ' ' << val_pts_all[i];
    }
    std::cout << std::endl;
   #endif
    // Unset the predicate for those selected as validation points
    turn_predicate_on(val_pts_all);
  }

  // distribute updated predicate
  if (!distribute_predicate(appDataLoc)) {
    return false;
  }
  clear_intermediate_info();

  return true;
}

template <typename FPTypeValue>
std::array<FPTypeValue, 2> VPCollector<FPTypeValue>::sum_sqdev(
  const FPTypeValue* phy_out,
  const FPTypeValue* sur_out,
  const StepValPoints<FPTypeValue>& step_valpoints,
  const std::array<FPTypeValue, 2>& avg)
{
  std::array<FPTypeValue, 2> sum{static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)};

  size_t k = 0ul;
  for (auto& pt: step_valpoints.val_pts()) {
    for (; k < pt.m_pos; ++k) {
      sum[0] += std::abs(phy_out[k] - sur_out[k]);
    }
    k++;
    auto diff = std::abs(pt.m_o_phy - pt.m_o_sur) - avg[1];
    sum[1] += diff*diff;
  }
  for (; k < m_num_pred_loc; ++k) {
    sum[0] += std::abs(phy_out[k] - sur_out[k]);
  }

  return sum;
}

#if __STANDALONE_TEST__
template <typename FPTypeValue>
void VPCollector<FPTypeValue>::backup_surrogate_outs(
  const FPTypeValue* sur_out,
  StepValPoints<FPTypeValue>& step_valpoints)
{
  m_sur.resize(1u, nullptr);
#if __STANDALONE_TEST__
  m_sur[0] = ams::ResourceManager::allocate<FPTypeValue>(m_num_pred_loc, m_appDataLoc);
#endif // __STANDALONE_TEST__
  std::memcpy(m_sur[0], sur_out, m_num_pred_loc);

  for (size_t i = 0ul; i < m_num_pred_loc; ++i) {
    if (m_pred_loc_new[i] != m_pred_loc[i]) {
      step_valpoints.add_pt(ValPoint<FPTypeValue>(i, sur_out[i]));
    }
  }
  //step_valpoints.sort_points();
  step_valpoints.inc_step();
}

template <typename FPTypeValue>
std::array<FPTypeValue, 2> VPCollector<FPTypeValue>::backup_validation_outs(
  const FPTypeValue* phy_out,
  StepValPoints<FPTypeValue>& step_valpoints)
{
  std::array<FPTypeValue, 2> sum {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)};

  size_t k = 0ul;
  for (auto& pt: step_valpoints.val_pts()) {
    for (; k < pt.m_pos; ++k) {
      sum[0] += std::abs(phy_out[k] - m_sur[0][k]);
    }
    k++;
    pt.m_o_phy = phy_out[pt.m_pos];
    sum[1] += std::abs(pt.m_o_phy - pt.m_o_sur);
  }

  for (; k < m_num_pred_loc; ++k) {
    sum[0] += std::abs(phy_out[k] - m_sur[0][k]);
  }

  return sum;
}

template <typename FPTypeValue>
ValStats<FPTypeValue> VPCollector<FPTypeValue>::get_error_stats(
  const FPTypeValue* phy_out,
  StepValPoints<FPTypeValue>& step_valpoints)
{
  const auto err_sum_loc = backup_validation_outs(phy_out, step_valpoints);
  std::array<FPTypeValue, 2> err_sum_glo {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)};
  std::array<FPTypeValue, 2> err_avg_glo {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)};

  std::array<unsigned, 2> err_cnt_loc {static_cast<unsigned>(m_num_pred_loc - step_valpoints.get_num_points()),
                                       static_cast<unsigned>(step_valpoints.get_num_points())};
  std::array<unsigned, 2> err_cnt_glo {0u, 0u};

  MPI_Allreduce(&err_cnt_loc[0], &err_cnt_glo[0], 2, MPI_UNSIGNED, MPI_SUM, m_comm);
  MPI_Datatype mpi_dtype = MPI_FLOAT;
  if (std::is_same<FPTypeValue, double>::value) {
    mpi_dtype = MPI_DOUBLE;
  } else //if (std::is_same<FPTypeValue, FPTypeValue>::value)
  {
    mpi_dtype = MPI_FLOAT;
  }

  MPI_Allreduce(&err_sum_loc[0], &err_sum_glo[0], 2, mpi_dtype, MPI_SUM, m_comm);
  err_avg_glo[0] = err_sum_glo[0] / static_cast<FPTypeValue>(err_cnt_glo[0]);
  err_avg_glo[1] = err_sum_glo[1] / static_cast<FPTypeValue>(err_cnt_glo[1]);

  const auto err_var_loc = sum_sqdev(phy_out, m_sur[0], step_valpoints, err_avg_glo);
  //err_var_loc[0] /= static_cast<FPTypeValue>(err_cnt_glo[0]);
  //err_var_loc[1] /= static_cast<FPTypeValue>(err_cnt_glo[1]);

  std::array<FPTypeValue, 2> err_var_glo{0,0};
  MPI_Allreduce(&err_var_loc[0], &err_var_glo[0], 2, mpi_dtype, MPI_SUM, m_comm);
  err_var_glo[0] /= static_cast<FPTypeValue>(err_cnt_glo[0]);
  err_var_glo[1] /= static_cast<FPTypeValue>(err_cnt_glo[1]);

  return ValStats<FPTypeValue>(err_cnt_glo, err_avg_glo, err_var_glo);
}
#endif // __STANDALONE_TEST__


template <typename FPTypeValue>
void VPCollector<FPTypeValue>::backup_surrogate_outs(
  const std::vector<FPTypeValue*> sur_out,
  std::vector<StepValPoints<FPTypeValue>>& step_valpoints)
{
  const size_t dim = sur_out.size();
  step_valpoints.clear();
  step_valpoints.resize(dim);

  m_sur.resize(dim);
  for (size_t j = 0ul; j < dim; ++j) {
#if __STANDALONE_TEST__
    m_sur[j] = ams::ResourceManager::allocate<FPTypeValue>(m_num_pred_loc, m_appDataLoc);
#else
    auto &rm_a = ams::ResourceManager::getInstance();
    m_sur[j] = rm_a.allocate<FPTypeValue>(m_num_pred_loc, m_appDataLoc);
#endif // __STANDALONE_TEST__
    std::memcpy(m_sur[j], sur_out[j], m_num_pred_loc);
  }

  for (size_t i = 0ul; i < m_num_pred_loc; ++i) {
    if (m_pred_loc_new[i] != m_pred_loc[i]) {
      for (size_t j = 0ul; j < dim; ++j) {
        step_valpoints[j].add_pt(ValPoint<FPTypeValue>(i, sur_out[j][i]));
      }
    }
  }
  for (size_t j = 0ul; j < dim; ++j) {
    //step_valpoints[j].sort_points();
    step_valpoints[j].inc_step();
  }
}

template <typename FPTypeValue>
std::vector<std::array<FPTypeValue, 2>> VPCollector<FPTypeValue>::backup_validation_outs(
  const std::vector<FPTypeValue*> phy_out,
  std::vector<StepValPoints<FPTypeValue>>& step_valpoints)
{
  const size_t dim = step_valpoints.size();

  std::vector<std::array<FPTypeValue, 2>> sum(
    dim,
    {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)});

  if (phy_out.size() != dim) {
    // exception
    std::cerr << "Invalud data dimension!" << std::endl;
    return sum;
  }

  for (size_t j = 0ul; j < dim; ++j) {
    size_t k = 0ul;
    for (auto& pt: step_valpoints[j].val_pts()) {
      for ( ; k < pt.m_pos; ++k) {
        sum[j][0] += std::abs(phy_out[j][k] - m_sur[j][k]);
      }
      k++;
      pt.m_o_phy = phy_out[j][pt.m_pos];
      sum[j][1] += std::abs(pt.m_o_phy - pt.m_o_sur);
    }
    for ( ; k < m_num_pred_loc; ++k) {
      sum[j][0] += std::abs(phy_out[j][k] - m_sur[j][k]);
    }
  }
  return sum;
}


template <typename FPTypeValue>
std::vector<ValStats<FPTypeValue>> VPCollector<FPTypeValue>::get_error_stats(
  const std::vector<FPTypeValue*> phy_out,
  std::vector<StepValPoints<FPTypeValue>>& step_valpoints)
{
  const auto err_sum_loc = backup_validation_outs(phy_out, step_valpoints);
  const size_t dim = err_sum_loc.size();
  std::vector<std::array<FPTypeValue, 2>> err_sum_glo(dim, {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)});
  std::vector<std::array<FPTypeValue, 2>> err_avg_glo(dim, {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)});
  std::vector<std::array<FPTypeValue, 2>> err_var_glo(dim, {static_cast<FPTypeValue>(0), static_cast<FPTypeValue>(0)});
  std::vector<ValStats<FPTypeValue>> stats(dim);

  std::array<unsigned, 2>  err_cnt_loc {m_num_pred_loc - step_valpoints.at(0).get_num_points(),
                                        step_valpoints.at(0).get_num_points()};
  std::array<unsigned, 2>  err_cnt_glo {0u, 0u};

  MPI_Allreduce(&err_cnt_loc[0], &err_cnt_glo[0], 2, MPI_UNSIGNED, MPI_SUM, m_comm);

  MPI_Datatype mpi_dtype = MPI_FLOAT;
  if (std::is_same<FPTypeValue, double>::value) {
    mpi_dtype = MPI_DOUBLE;
  } else //if (std::is_same<FPTypeValue, float>::value)
  {
    mpi_dtype = MPI_FLOAT;
  }

  for (size_t j = 0ul; j < dim; ++j) {
    MPI_Allreduce(&err_sum_loc[j][0], &err_sum_glo[j][0], 2, mpi_dtype, MPI_SUM, m_comm);
    err_avg_glo[j][0] = err_sum_glo[j][0] / static_cast<FPTypeValue>(err_cnt_glo[0]);
    err_avg_glo[j][1] = err_sum_glo[j][1] / static_cast<FPTypeValue>(err_cnt_glo[1]);
    std::array<FPTypeValue, 2> err_var_loc =
      sum_sqdev(phy_out[j], step_valpoints[j], err_avg_glo[j], j);
    //err_var_loc[0] /= static_cast<FPTypeValue>(err_cnt_glo[0]);
    //err_var_loc[1] /= static_cast<FPTypeValue>(err_cnt_glo[1]);
    MPI_Allreduce(&err_var_loc, &err_var_glo[j], 1, mpi_dtype, MPI_SUM, m_comm);
    err_var_glo[j][0] /= static_cast<FPTypeValue>(err_cnt_glo[0]);
    err_var_glo[j][1] /= static_cast<FPTypeValue>(err_cnt_glo[1]);
    stats[j] = ValStats<FPTypeValue>(err_cnt_glo, err_avg_glo[j], err_var_glo[j]);
  }

  return stats;
}


}  // namespace ams

#endif // __AMS_VALIDATOR_HPP__
