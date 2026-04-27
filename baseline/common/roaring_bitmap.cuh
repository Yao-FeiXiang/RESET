/**
 * @file roaring_bitmap.cuh
 * @brief 动态压缩Roaring位图 - 只分配实际使用的容器，避免内存浪费
 */

#ifndef ROARING_BITMAP_CUH
#define ROARING_BITMAP_CUH

#include <cuda_runtime.h>

#include <vector>

#include "utils.cuh"

class RoaringBitmap {
 public:
  static constexpr int BITS_PER_WORD = 64;
  static constexpr int LOG_BITS_PER_WORD = 6;
  static constexpr int MASK_BITS_PER_WORD = 0x3F;

  // 让我们重新设计：轻量级Roaring，每个节点只存储存在的word
  static constexpr int CONTAINER_BITS = 22;  // 高1位选容器，低21位选位
  static constexpr int MAX_KEY_LOW = (1 << CONTAINER_BITS);
  static constexpr int WORDS_PER_CONTAINER =
      MAX_KEY_LOW / BITS_PER_WORD;  // 65536

  /**
   * @brief 构造函数 - 动态计算所需内存
   * @param num_nodes 节点数量
   * @param csr_offsets CSR偏移数组(主机端)
   * @param csr_cols CSR列数组(主机端)
   */
  RoaringBitmap(int num_nodes, const std::vector<int>& csr_offsets,
                const std::vector<int>& csr_cols);

  /**
   * @brief 析构函数,释放设备内存
   */
  ~RoaringBitmap();

  // 禁用拷贝
  RoaringBitmap(const RoaringBitmap&) = delete;
  RoaringBitmap& operator=(const RoaringBitmap&) = delete;

  /**
   * @brief 批量插入所有节点的邻接点 - 实际构建位图
   * @return 总是返回0(不会失败)
   */
  int bulk_insert();

  /**
   * @brief 查询：检查一个键是否在指定节点的位图中(设备端调用)
   * @param node_id 查询节点ID
   * @param key 要查询的键
   * @return 是否存在
   */
  __device__ bool contains(int node_id, int key) const {
    const int remapped_key = d_id_remap_[key];
    const int word_idx = remapped_key >> LOG_BITS_PER_WORD;
    const int bit_pos = remapped_key & MASK_BITS_PER_WORD;
    const uint64_t bit_mask = 1ULL << bit_pos;

    const int start = d_node_start_[node_id];
    const int end = d_node_start_[node_id + 1];
    const int num_words = end - start;

    if (num_words == 0) return false;

    // 二分查找word_idx
    int left = 0, right = num_words - 1;
    while (left <= right) {
      const int mid = (left + right) / 2;
      const int mid_word_idx = d_word_index_[start + mid];

      if (mid_word_idx == word_idx) {
        const uint64_t word_data = d_word_data_[start + mid];
        return (word_data & bit_mask) != 0;
      } else if (mid_word_idx < word_idx) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    return false;
  }

  /**
   * @brief 获取总容量(字数)
   * @return 总容量
   */
  size_t total_words() const { return total_words_; }

 private:
  int* d_node_start_;      // 每个节点在word数组中的起始偏移
  int* d_word_index_;      // word索引数组
  uint64_t* d_word_data_;  // word位图数据数组
  int* d_id_remap_;        // 节点ID重映射表
  size_t total_words_;     // 总word数
  int num_nodes_;          // 节点数量

  // 保存原始CSR数据引用，用于bulk_insert
  const std::vector<int>& csr_offsets_;
  const std::vector<int>& csr_cols_;

  // 主机端暂存 - 用于构建阶段
  std::vector<int> host_node_start_;
  std::vector<int> host_word_index_;
  std::vector<uint64_t> host_word_data_;
  std::vector<int> host_id_remap_;  // 主机端ID重映射表
};

#endif  // ROARING_BITMAP_CUH
