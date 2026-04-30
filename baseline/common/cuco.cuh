/**
 * @file cuco.cuh
 * @brief cuCollections static_set_ref 实现
 */

#ifndef CUCO_CUH
#define CUCO_CUH

#include <cuda_runtime.h>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_set_ref.cuh>
#include <limits>
#include <vector>

#include "utils.cuh"

class CucoHash {
 public:
  static constexpr int EMPTY_KEY = std::numeric_limits<int>::max();
  static constexpr int CG_SIZE = 1;
  static constexpr int BUCKET_SIZE = 1;

  using key_type = int;
  using hasher = cuco::murmurhash3_fmix_32<key_type>;
  using probing_scheme = cuco::linear_probing<CG_SIZE, hasher>;
  using storage_ref_type = cuco::bucket_storage_ref<key_type, BUCKET_SIZE,
                                                    cuco::extent<std::size_t>>;
  using ref_type = cuco::static_set_ref<key_type, cuda::thread_scope_device,
                                        cuda::std::equal_to<key_type>,
                                        probing_scheme, storage_ref_type>;

  CucoHash(int num_nodes, const std::vector<int>& degrees,
           float load_factor = 2.0f);
  ~CucoHash();

  CucoHash(const CucoHash&) = delete;
  CucoHash& operator=(const CucoHash&) = delete;

  void bulk_insert(const std::vector<int>& csr_offsets,
                   const std::vector<int>& csr_cols,
                   cudaStream_t stream = nullptr);

  int* get_device_table() const { return d_slots_; }
  long long* get_device_offsets() const { return d_offsets_; }
  long long get_total_capacity() const { return total_capacity_; }
  int get_num_nodes() const { return num_nodes_; }

 private:
  static std::size_t next_power_of_two(std::size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
  }

  int num_nodes_;
  float load_factor_;
  long long total_capacity_;

  std::vector<long long> h_offsets_;
  int* d_slots_;
  long long* d_offsets_;
};

__device__ __forceinline__ bool cuco_contains(int node_id, int key, int* slots,
                                              long long* offsets,
                                              int empty_key) {
  long long start = offsets[node_id];
  int capacity = static_cast<int>(offsets[node_id + 1] - start);

  if (capacity == 0) return false;

  cuco::extent<std::size_t> extent(capacity / CucoHash::BUCKET_SIZE);
  CucoHash::storage_ref_type storage_ref(extent, slots + start);

  CucoHash::probing_scheme probing;
  cuda::std::equal_to<int> equal;

  CucoHash::ref_type ref(cuco::empty_key<int>{empty_key}, equal, probing,
                         cuco::cuda_thread_scope<cuda::thread_scope_device>{}, storage_ref);

  auto contains_ref = ref.rebind_operators(cuco::op::contains);

  return contains_ref.contains(key);
}

#endif  // CUCO_CUH
