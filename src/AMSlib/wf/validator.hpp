#ifndef __AMS_VALIDATOR_HPP__
#define __AMS_VALIDATOR_HPP__
#include <iostream>
#include <string>
#include <cstring> // memcpy
#include <set>
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
  bool count_local_vals(const std::vector<size_t>& val_pts,
                        std::vector<unsigned>& num_chosen_per_rank);
  bool turn_predicate_on(const unsigned my_val_pt_cnt);
  bool* distribute_predicate(const std::vector<unsigned>& local_val_pt_cnts,
                             const AMSResourceType appDataLoc);
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
  /// Local predicate array
  const bool* m_pred_loc;
  /// indices of positive local predicates
  std::vector<size_t> m_pred_loc_pos;
  bool* m_pred_loc_new;
  AMSResourceType m_appDataLoc;
  std::vector<FPTypeValue*> m_sur; ///< surrogate output backup

  // For root rank
  /// The total number of predicates
  size_t m_tot_num_preds;
  /// The total number of positive predicates
  size_t m_tot_num_preds_pos;
  /// number of predicates and the positive ones of each rank
  std::vector<unsigned> m_num_pred_all;
  /// displacement of positive predicates across ranks
  std::vector<size_t> m_displs_pos;
  std::default_random_engine m_rng;
};

template <typename FPTypeValue>
void VPCollector<FPTypeValue>::clear_intermediate_info()
{
  m_tot_num_preds = 0u;
  m_tot_num_preds_pos = 0u;
  m_num_pred_all.clear();
  m_displs_pos.clear();
}

template <typename FPTypeValue>
VPCollector<FPTypeValue>::VPCollector(unsigned seed, MPI_Comm comm)
: m_comm(comm), m_num_pred_loc(0u), m_pred_loc(nullptr), m_pred_loc_new(nullptr),
  m_tot_num_preds(0u), m_tot_num_preds_pos(0u)
{
  if (comm == MPI_COMM_NULL) {
    m_rank = 0;
    m_num_ranks = 1;
  } else {
    MPI_Comm_rank(comm, &m_rank);
    MPI_Comm_size(comm, &m_num_ranks);
  }
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
  int rc = MPI_SUCCESS;

  m_pred_loc_pos.clear();
  m_pred_loc_pos.reserve(m_num_pred_loc);

  for (size_t i = 0u; i < m_num_pred_loc; ++i) {
    if (m_pred_loc[i]) m_pred_loc_pos.push_back(i);
  }

  std::vector<unsigned> cnt_loc = {static_cast<unsigned>(m_num_pred_loc),
                                   static_cast<unsigned>(m_pred_loc_pos.size())};

  m_num_pred_all.clear();
  if (m_rank == 0) {
    m_num_pred_all.resize(m_num_ranks*2);
  }
  // Gather the data sizes (i.e., the number of items) from each rank
  rc = MPI_Gather(reinterpret_cast<const void*>(cnt_loc.data()),
                  2,
                  MPI_UNSIGNED,
                  reinterpret_cast<void*>(m_num_pred_all.data()),
                  2,
                  MPI_UNSIGNED,
                  0,
                  m_comm);

  if (rc != MPI_SUCCESS) {
    if (m_rank == 0) {
      std::cerr << "MPI_Gather() in gather_predicate() failed with code ("
                << rc << ")" << std::endl;
    }
    return rc;
  }

  m_displs_pos.clear();

  if (m_rank == 0) {
    m_displs_pos.resize(m_num_ranks);

    size_t offset = 0u;
    size_t offset_pos = 0u;
    auto it = m_num_pred_all.cbegin();
    for (int i = 0; i < m_num_ranks; ++i) {
      m_displs_pos[i] = offset_pos;
      offset += static_cast<size_t>(*it++);
      offset_pos += static_cast<size_t>(*it++);
    }
    m_tot_num_preds = offset;
    m_tot_num_preds_pos = offset_pos;
  }
  return MPI_SUCCESS;
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

/** Randonly choose the points to run physics on, out of those accepted with
    the surrogate.
    `k_v': the minimum number of new physics evaluation points per rank.
           By default, it is 0. Each rank will have k_v or k_v+1 extra physic
           evaluation points. */

template <typename FPTypeValue>
std::vector<size_t> VPCollector<FPTypeValue>::pick_val_pts(unsigned k_v)
{
  if (m_tot_num_preds_pos < 1u) {
    return std::vector<size_t>{};
  }

  const size_t num_vals = pick_num_val_pts(m_tot_num_preds_pos,
                                           m_tot_num_preds - m_tot_num_preds_pos,
                                           k_v);

  std::uniform_int_distribution<size_t> dist(0u, m_tot_num_preds_pos-1);

  std::set<size_t> chosen;
  while (chosen.size() < num_vals) {
    chosen.insert(dist(m_rng));
  }

  return std::vector<size_t>(chosen.cbegin(), chosen.cend());
}

/** The root rank could have translated the indices of chosen predicates to the
 *  local indices of owner ranks and then send the indices back to each rank.
 *  However, as each rank can own a various number of predicates chosen to
 *  flip, it would require sending the number of indices for each rank to
 *  receive in advance. To save MPI communication, we simply send the count to
 *  flip to each rank and let each rank locally determine which predicates to
 *  flip as many as the count it receives. Here, we prepare the count for each
 *  rank. */
template <typename FPTypeValue>
bool VPCollector<FPTypeValue>::count_local_vals(const std::vector<size_t>& val_pts,
                                                std::vector<unsigned>& num_chosen_per_rank)
{
  num_chosen_per_rank.assign(m_num_ranks, 0u);
  int cur_rank = 0;

  for (auto idx: val_pts) {
    while (idx >= m_displs_pos[cur_rank+1]) {
      ++ cur_rank;
      if (cur_rank >= m_num_ranks) {
        std::cerr << "Invalid predicate index generated for error analysis!" << std::endl;
        return false;
      }
    }
    num_chosen_per_rank[cur_rank]++;
  }
  return true;
}

template <typename FPTypeValue>
bool VPCollector<FPTypeValue>::turn_predicate_on(const unsigned my_val_pt_cnt)
{
  std::memcpy(m_pred_loc_new, m_pred_loc, m_num_pred_loc*sizeof(bool));

  if (static_cast<size_t>(my_val_pt_cnt) > m_pred_loc_pos.size()) {
    std::cerr << "Invalid number of positive predicates computed for error analysis!" << std::endl;
    return false;
  }
  if (my_val_pt_cnt == 0u) {
    return true;
  }

  auto pred_idx = m_pred_loc_pos;

  std::shuffle(pred_idx.begin(), pred_idx.end(), m_rng);

  for (unsigned i = 0u; i < my_val_pt_cnt; ++i) {
  #if DEBUG
    if (!m_pred_loc_new[pred_idx[i]]) {
      std::cerr << "Incorrect predicate chosen to flip for error analysis!" << std::endl;
    }
  #endif
    m_pred_loc_new[pred_idx[i]] = false;
  }

  return true;
}

template <typename FPTypeValue>
bool* VPCollector<FPTypeValue>::distribute_predicate(
    const std::vector<unsigned>& local_val_pt_cnts,
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

  unsigned my_val_pt_cnt = 0u;
  rc = MPI_Scatter(
         reinterpret_cast<const void*>(local_val_pt_cnts.data()),
         1,
         MPI_UNSIGNED,
         reinterpret_cast<void*>(&my_val_pt_cnt),
         1,
         MPI_UNSIGNED,
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
    m_pred_loc_new = nullptr;

    return nullptr;
  }
  clear_intermediate_info();

  // Unset the predicate for those selected as validation points
  if (!turn_predicate_on(my_val_pt_cnt)) {
    std::cerr << "Failed to turn predicates on for extra validation!" << std::endl;
#if __STANDALONE_TEST__
    ams::ResourceManager::deallocate(m_pred_loc_new, appDataLoc);
#else
    auto &rm_d = ams::ResourceManager::getInstance();
    rm_d.deallocate(m_pred_loc_new, appDataLoc);
#endif // __STANDALONE_TEST__
    m_pred_loc_new = nullptr;
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

  std::vector<unsigned> local_val_pt_cnts;
  if (m_rank == 0) {
    // indices of predicates to flip for enabling validation
    std::vector<size_t> pred_idx = pick_val_pts(k_v);
    if (!count_local_vals(pred_idx, local_val_pt_cnts)) {
      return false;
    }
  }

  // distribute updated predicate
  if (!distribute_predicate(local_val_pt_cnts, appDataLoc)) {
    return false;
  }

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

#if DEBUG
  if (m_num_pred_loc <= step_valpoints.at(0).get_num_points()) {
    std::cerr << "Incorrect number of validation points!" << std::endl;
  }
#endif

  std::array<unsigned, 2>  err_cnt_loc {static_cast<unsigned>(m_num_pred_loc - step_valpoints.at(0).get_num_points()),
                                        static_cast<unsigned>(step_valpoints.at(0).get_num_points())};
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
      sum_sqdev(phy_out[j], m_sur[j], step_valpoints[j], err_avg_glo[j]);
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
