/**
 * @file flat_cuckoo_hash.cuh
 * @brief 扁平化布谷鸟哈希 - 每个节点独立哈希表,整体连续存储,通过offset访问
 *
 * 设计特点：
 * - 每个节点单独构建布谷鸟哈希表,保存该节点的所有邻接点
 * - 所有哈希表连续存储在一块大内存中(扁平化存储)
 * - 通过offset数组定位每个节点哈希表的起始位置
 * - 支持任意节点数量和任意度数分布
 * - 三哈希布谷鸟策略 + stash,降低冲突概率,保证所有元素可找到
 *
 * 不变性保证：
 * - 插入过程中每次踢出都保持：每个key始终在它自己的三个候选位置之一
 * - 查询只需要检查三个候选位置 + stash,一定能找到已插入的key
 */

#ifndef FLAT_CUCKOO_HASH_CUH
#define FLAT_CUCKOO_HASH_CUH

#include <cuda_runtime.h>

#include <limits>
#include <vector>

#include "utils.cuh"

/**
 * @brief 扁平化布谷鸟哈希表
 *
 * 存储结构：
 *   [offset[0], offset[1], ..., offset[num_nodes+1]]
 *   [stash_count[0], stash_count[1], ..., stash_count[num_nodes]]
 *   [stash_storage...]
 *   [node 0's hash table ...]
 *   [node 1's hash table ...]
 *   ...
 *   [node n's hash table ...]
 *
 *   offset[i] 表示第i个节点哈希表的起始位置
 *   offset[i+1] - offset[i] 就是第i个节点哈希表的容量
 *   capacity[i] = (degree / load_factor) 向上对齐到2的幂次
 *   每个节点有固定大小的stash存储无法放入主表的元素
 */
class FlatCuckooHash {
 public:
  static constexpr int EMPTY_KEY = std::numeric_limits<int>::max();
  static constexpr int MAX_ITERATIONS = 6000;  // 最大踢出次数
  static constexpr int STASH_SIZE = 16;        // 每个节点的stash大小

  /**
   * @brief 构造函数
   * @param num_nodes 节点数量
   * @param degrees 每个节点的度数
   * @param load_factor 负载因子
   */
  FlatCuckooHash(int num_nodes, const std::vector<int>& degrees,
                 float load_factor);

  /**
   * @brief 析构函数,释放设备内存
   */
  ~FlatCuckooHash();

  // 禁用拷贝
  FlatCuckooHash(const FlatCuckooHash&) = delete;
  FlatCuckooHash& operator=(const FlatCuckooHash&) = delete;

  /**
   * @brief 批量插入所有节点的邻接点
   * @param csr_offsets CSR偏移数组(主机端)
   * @param csr_cols CSR列数组(主机端)
   * @return 插入失败的总个数(stash也放不下了)
   */
  int bulk_insert(const std::vector<int>& csr_offsets,
                  const std::vector<int>& csr_cols);

  /**
   * @brief 查询：检查一个键是否在指定节点的哈希表中(设备端调用)
   * @param node_id 查询节点ID
   * @param key 要查询的键
   * @param table 设备端哈希表指针
   * @param offsets 设备端偏移指针
   * @param stash_starts 设备端stash偏移
   * @param stash_data 设备端stash数据
   * @return true表示存在
   */
  __device__ __forceinline__ bool contains(int node_id, int key, int* table,
                                           long long* offsets,
                                           long long* stash_starts,
                                           int* stash_data) const {
    long long start = offsets[node_id];
    int capacity = static_cast<int>(offsets[node_id + 1] - start);

    // ✅ 使用与插入完全一致的标准哈希函数
    int h1 = standard_hash(key, node_id, capacity);
    int h2 = standard_hash2(key, node_id, capacity);

    long long pos1 = start + h1;
    long long pos2 = start + h2;

    if (table[pos1] == key) return true;
    if (table[pos2] == key) return true;

    // 主表没找到,检查stash
    long long stash_start = node_id * STASH_SIZE;
    int stash_count = static_cast<int>(stash_starts[node_id]);
    for (int i = 0; i < stash_count; i++) {
      if (stash_data[stash_start + i] == key) {
        return true;
      }
    }

    return false;
  }

  // 获取设备端指针
  int* get_device_table() const { return d_table_; }
  long long* get_device_offsets() const { return d_offsets_; }
  long long* get_device_stash_starts() const { return d_stash_starts_; }
  int* get_device_stash_data() const { return d_stash_data_; }

  // 获取失败计数
  int get_failed_count() const { return failed_count_; }

  // 获取总容量
  long long get_total_capacity() const { return total_capacity_; }

 private:
  int num_nodes_;
  long long total_capacity_;
  long long total_stash_capacity_;
  std::vector<long long> h_offsets_;       // 主机端偏移数组
  std::vector<long long> h_stash_starts_;  // 主机端stash偏移

  int* d_table_ = nullptr;               // 设备端扁平化哈希表
  long long* d_offsets_ = nullptr;       // 设备端偏移数组
  long long* d_stash_starts_ = nullptr;  // 设备端stash偏移(stash_count每个node)
  int* d_stash_data_ = nullptr;          // 设备端stash数据
  int* d_failed_ = nullptr;              // 设备端失败计数
  int failed_count_;                     // 主机端保存的失败计数
};

#endif  // FLAT_CUCKOO_HASH_CUH
