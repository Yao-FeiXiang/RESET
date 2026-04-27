/**
 * @file cuco.cuh
 */

#ifndef CUCO_CUH
#define CUCO_CUH

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "utils.cuh"

/**
 * @brief cuco哈希表(手动实现线性探测)
 *
 * 存储结构：
 *   [offset[0], offset[1], ..., offset[num_nodes+1]]
 *   [node 0's hash table slots ...]
 *   [node 1's hash table slots ...]
 *   ...
 *   [node n's hash table slots ...]
 *
 *   offset[i] 表示第i个节点哈希表的起始位置(单位:slot)
 *   offset[i+1] - offset[i] 就是第i个节点哈希表的容量(slots数)
 *   capacity[i] = next_power_of_two(degree / load_factor)
 */
class CucoHash {
 public:
  static constexpr int EMPTY_KEY = std::numeric_limits<int>::max();

  /**
   * @brief 构造函数
   * @param num_nodes 节点数量
   * @param degrees 每个节点的度数(主机端)
   * @param load_factor 负载因子(默认2.0)
   */
  CucoHash(int num_nodes, const std::vector<int>& degrees,
           float load_factor = 2.0f);

  /**
   * @brief 析构函数,释放设备内存
   */
  ~CucoHash();

  // 禁用拷贝
  CucoHash(const CucoHash&) = delete;
  CucoHash& operator=(const CucoHash&) = delete;

  /**
   * @brief 批量插入所有节点的邻接点
   * @param csr_offsets CSR偏移数组(主机端)
   * @param csr_cols CSR列数组(主机端)
   * @param stream CUDA流
   */
  void bulk_insert(const std::vector<int>& csr_offsets,
                   const std::vector<int>& csr_cols,
                   cudaStream_t stream = nullptr);

  /**
   * @brief 获取设备端扁平哈希表指针
   * @return 设备端指针
   */
  int* get_device_table() const { return d_slots_; }

  /**
   * @brief 获取设备端偏移数组指针
   * @return 设备端offset指针
   */
  long long* get_device_offsets() const { return d_offsets_; }

  /**
   * @brief 获取总容量
   * @return 总slot数量
   */
  long long get_total_capacity() const { return total_capacity_; }

  /**
   * @brief 获取节点数量
   * @return 节点数
   */
  int get_num_nodes() const { return num_nodes_; }

 private:
  /**
   * @brief 向上取整到2的幂次
   */
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
  long long total_capacity_;  // 总slot数

  std::vector<long long> h_offsets_;  // 主机端偏移数组

  int* d_slots_ = nullptr;          // 设备端slot数组
  long long* d_offsets_ = nullptr;  // 设备端偏移数组
};

/**
 * @brief 设备端查询辅助函数：检查key是否在指定节点的哈希表中
 *
 * @param node_id 节点ID
 * @param key 要查询的键
 * @param slots 设备端slot数组指针
 * @param offsets 设备端offset数组
 * @param empty_key 空键标记
 * @return true表示存在
 *
 */
__device__ __forceinline__ std::uint32_t murmur3_32(int key) {
  std::uint32_t h = static_cast<std::uint32_t>(key);
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

__device__ __forceinline__ bool cuco_contains(int node_id, int key, int* slots,
                                              long long* offsets,
                                              int empty_key) {
  long long start = offsets[node_id];
  int capacity = static_cast<int>(offsets[node_id + 1] - start);

  if (capacity == 0) return false;

  // 线性探测 - 与cuCollections保持一致的行为
  std::uint32_t hash = murmur3_32(key);
  int probe_count = 0;
  int mask = capacity - 1;  // capacity是2的幂

  while (probe_count < capacity) {
    int pos = (hash + probe_count) & mask;
    int current = slots[start + pos];

    if (current == empty_key) {
      return false;
    }
    if (current == key) {
      return true;
    }
    probe_count++;
  }
  return false;
}

#endif  // CUCO_CUH
