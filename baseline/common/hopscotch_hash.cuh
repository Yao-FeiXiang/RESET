/**
 * @file hopscotch_hash.cuh
 * @brief 扁平化跳房子哈希 - 每个节点独立哈希表,整体连续存储,通过offset访问
 *
 * 设计特点：
 * - 每个节点单独构建跳房子哈希表,保存该节点的所有邻接点
 * - 所有哈希表连续存储在一块大内存中(扁平化存储)
 * - 通过offset数组定位每个节点哈希表的起始位置
 * - 支持任意节点数量和任意度数分布
 * - H=32固定邻域大小,优秀的缓存局部性
 *
 * 算法特点：
 * - 每个key限制在home bucket的H=32范围内
 * - 使用位图标记哪些位置被占据
 * - 通过移动空闲位置到邻域解决冲突
 * - 查询时只需要检查位图中标记的位置,缓存友好
 */

#ifndef HOPSCOTCH_HASH_CUH
#define HOPSCOTCH_HASH_CUH

#include <cuda_runtime.h>

#include <limits>
#include <vector>

#include "utils.cuh"

/**
 * @brief 扁平化跳房子哈希表
 *
 * 存储结构：
 *   [offset[0], offset[1], ..., offset[num_nodes+1]]
 *   [table_storage...]
 *   [bitmap_storage...]
 *
 *   offset[i] 表示第i个节点哈希表的起始位置
 *   offset[i+1] - offset[i] 就是第i个节点哈希表的容量
 *   capacity[i] = (degree / load_factor) 向上对齐
 *   每个bucket对应一个32位位图,标记邻域中哪些位置被占据
 */
class HopscotchHash {
 public:
  static constexpr int EMPTY_KEY = std::numeric_limits<int>::max();
  static constexpr int H = 32;  // 邻域大小

  /**
   * @brief 构造函数
   * @param num_nodes 节点数量
   * @param degrees 每个节点的度数
   * @param load_factor 负载因子
   */
  HopscotchHash(int num_nodes, const std::vector<int>& degrees,
                float load_factor);

  /**
   * @brief 析构函数,释放设备内存
   */
  ~HopscotchHash();

  // 禁用拷贝
  HopscotchHash(const HopscotchHash&) = delete;
  HopscotchHash& operator=(const HopscotchHash&) = delete;

  /**
   * @brief 批量插入所有节点的邻接点
   * @param csr_offsets CSR偏移数组(主机端)
   * @param csr_cols CSR列数组(主机端)
   * @return 插入失败的总个数
   */
  int bulk_insert(const std::vector<int>& csr_offsets,
                  const std::vector<int>& csr_cols);

  /**
   * @brief 查询：检查一个键是否在指定节点的哈希表中(设备端调用)
   * @param node_id 查询节点ID
   * @param key 要查询的键
   * @return 是否存在
   */
  __device__ bool contains(int node_id, int key) const {
    int node_start = d_offset_[node_id];
    //  ✔ 修正：总容量减去邻域大小H才是实际哈希表容量
    int total_with_h = d_offset_[node_id + 1] - node_start;
    int capacity = total_with_h - H;
    int home = standard_hash(key, node_id, capacity);

    uint32_t bm = d_bitmap_[node_start + home];

    while (bm != 0) {
      // 提取最低有效位
      int i = __ffs(bm) - 1;
      bm ^= (1U << i);
      //  ✔ 边界检查：确保位置在有效范围内
      if (home + i < total_with_h) {
        int pos = node_start + home + i;
        if (d_table_[pos] == key) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * @brief 获取总容量
   * @return 总容量
   */
  size_t total_capacity() const { return total_capacity_; }

  // 获取设备端指针
  int* get_device_offset() const { return d_offset_; }
  int* get_device_table() const { return d_table_; }
  int* get_device_bitmap() const { return d_bitmap_; }

 private:
  // Note: contains() 现在直接使用 standard_hash() from utils.cuh

  int* d_offset_;          // 每个节点的起始偏移
  int* d_table_;           // 所有哈希表存储 (扁平化连续)
  int* d_bitmap_;          // 所有位图存储 (每个bucket一个32位位图)
  size_t total_capacity_;  // 总容量
  int num_nodes_;          // 节点数量
};

#endif  // HOPSCOTCH_HASH_CUH
