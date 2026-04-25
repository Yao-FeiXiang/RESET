/**
 * @file roaring_bitmap.cuh
 * @brief 动态压缩Roaring位图 - 只分配实际使用的容器，避免内存浪费
 *
 * 优化设计：
 * - 原始问题：为每个节点分配所有可能的容器（即使大部分容器为空），导致内存爆炸
 * - 优化方案：每个节点只分配实际用到的high-bit容器，大幅节省内存
 *
 * 存储结构（设备端连续存储）：
 *   [node_offsets] 每个节点在container_index中的起始偏移 → 长度 num_nodes+1
 *   [container_index] 排序存储所有节点实际使用的容器编号 → 总长度 =
 * 所有节点实际容器数 [bitmap_storage] 每个容器对应的位数组（每个容器固定1024字
 * × 8字节 = 8KB）
 *
 * 内存节省：
 * - 对于平均度数d，每个节点平均只使用 ≈ ceil(d / 65536) 个容器
 * - 示例：100K节点，平均度数10 → 总容器≈ 100K × (10/65536) ≈ 16 容器 → 总位图
 * 16 × 8KB = 128KB！
 * - 原始版本：100K节点，max_key=100K → 每个节点分配2容器 → 总200K容器
 * → 1.6GB，差别巨大！
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
  static constexpr int CONTAINER_BITS = 16;
  static constexpr int MAX_KEY_LOW = (1 << CONTAINER_BITS);
  static constexpr int WORDS_PER_CONTAINER = MAX_KEY_LOW / BITS_PER_WORD;

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
    // Roaring位图映射
    const int container = key >> 16;   // 高16位是容器索引
    const int bit_pos = key & 0xFFFF;  // 低16位是位位置
    const int word_idx = bit_pos >> LOG_BITS_PER_WORD;
    const unsigned long long bit_mask = 1ULL << (bit_pos & MASK_BITS_PER_WORD);

    // 二分查找该容器是否存在于此节点
    const int node_start = d_node_offsets_[node_id];
    const int node_end = d_node_offsets_[node_id + 1];
    const int num_containers_node = node_end - node_start;

    if (num_containers_node == 0) return false;

    // 二分查找容器
    int left = 0, right = num_containers_node - 1;
    while (left <= right) {
      const int mid = (left + right) / 2;
      const int mid_container = d_container_index_[node_start + mid];

      if (mid_container == container) {
        // 找到容器！计算位图位置
        // 每个container对应WORDS_PER_CONTAINER个字
        // 使用size_t避免32位溢出
        const size_t bitmap_offset =
            (size_t)(node_start + mid) * (size_t)WORDS_PER_CONTAINER;
        return (d_bitmap_[bitmap_offset + word_idx] & bit_mask) != 0;
      } else if (mid_container < container) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    // 容器不存在，key一定不存在
    return false;
  }

  /**
   * @brief 获取总容量(字数)
   * @return 总容量
   */
  size_t total_words() const { return total_words_; }

 private:
  int* d_node_offsets_;  // 每个节点在container_index数组中的起始偏移 → 长度
                         // num_nodes+1
  int* d_container_index_;  // 排序存储所有节点实际使用的容器编号 → 长度
                            // total_containers
  unsigned long long* d_bitmap_;  // 每个容器对应的位数组 → 总字数 =
                                  // total_containers * WORDS_PER_CONTAINER
  size_t total_words_;  // 总bitmap字数 = total_containers * WORDS_PER_CONTAINER
  int num_nodes_;       // 节点数量
  int total_containers_;  // 所有节点总共使用的容器总数

  // 保存原始CSR数据引用，用于bulk_insert
  const std::vector<int>& csr_offsets_;
  const std::vector<int>& csr_cols_;

  // 主机端暂存 - 用于构建阶段
  std::vector<int> host_node_offsets_;
  std::vector<int> host_container_index_;
};

#endif  // ROARING_BITMAP_CUH
