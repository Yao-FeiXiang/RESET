/**
 * @file roaring_bitmap.cu
 * @brief 轻量级Roaring位图 - 针对SSS低度数图优化
 *
 */

#include <thrust/device_vector.h>

#include <algorithm>
#include <unordered_map>

#include "roaring_bitmap.cuh"
#include "utils.cuh"

/**
 * 构造函数：稀疏Roaring位图 - 主机端构建
 */
__host__ RoaringBitmap::RoaringBitmap(int num_nodes,
                                      const std::vector<int>& csr_offsets,
                                      const std::vector<int>& csr_cols)
    : num_nodes_(num_nodes), csr_offsets_(csr_offsets), csr_cols_(csr_cols) {
  // ========== 第一步：ID重映射 ==========
  std::vector<int> all_ids;
  all_ids.reserve(csr_cols.size() + num_nodes);

  for (int i = 0; i < num_nodes; i++) {
    all_ids.push_back(i);
  }
  for (int id : csr_cols) {
    all_ids.push_back(id);
  }

  std::sort(all_ids.begin(), all_ids.end());
  all_ids.erase(std::unique(all_ids.begin(), all_ids.end()), all_ids.end());

  int max_original_id = all_ids.back();
  host_id_remap_.resize(max_original_id + 1);
  for (int i = 0; i < all_ids.size(); i++) {
    host_id_remap_[all_ids[i]] = i;
  }

  // ========== 第二步：稀疏存储每个节点的邻接集 ==========
  host_node_start_.resize(num_nodes + 1);
  host_node_start_[0] = 0;

  for (int node_id = 0; node_id < num_nodes; node_id++) {
    const int start = csr_offsets[node_id];
    const int end = csr_offsets[node_id + 1];

    // 收集该节点所有邻居 -> word位图
    std::unordered_map<int, uint64_t> word_map;

    for (int i = start; i < end; i++) {
      const int original_key = csr_cols[i];
      const int remapped_key = host_id_remap_[original_key];
      const int word_idx = remapped_key >> LOG_BITS_PER_WORD;
      const int bit_pos = remapped_key & MASK_BITS_PER_WORD;

      word_map[word_idx] |= (1ULL << bit_pos);
    }

    // 排序并分别存储
    std::vector<std::pair<int, uint64_t>> sorted_words(word_map.begin(),
                                                       word_map.end());
    std::sort(sorted_words.begin(), sorted_words.end());

    for (const auto& p : sorted_words) {
      host_word_index_.push_back(p.first);
      host_word_data_.push_back(p.second);
    }

    host_node_start_[node_id + 1] = host_word_index_.size();
  }

  total_words_ = host_word_index_.size();

  const double avg_words_per_node =
      static_cast<double>(total_words_) / num_nodes;

  // printf(
  //     "[Roaring] 节点数: %d, 总word数: %zu, 平均每节点word数: %.3f\n"
  //     "[Roaring] 总内存: %.2f MB (node_start: %.2f MB, word_index: %.2f MB, "
  //     "word_data: %.2f MB, remap: %.2f MB)\n",
  //     num_nodes, total_words_, avg_words_per_node,
  //     static_cast<double>((num_nodes + 1) * sizeof(int) +
  //                         total_words_ * sizeof(int) +
  //                         total_words_ * sizeof(uint64_t) +
  //                         host_id_remap_.size() * sizeof(int)) /
  //         (1024.0 * 1024.0),
  //     static_cast<double>((num_nodes + 1) * sizeof(int)) / (1024.0 * 1024.0),
  //     static_cast<double>(total_words_ * sizeof(int)) / (1024.0 * 1024.0),
  //     static_cast<double>(total_words_ * sizeof(uint64_t)) / (1024.0 *
  //     1024.0), static_cast<double>(host_id_remap_.size() * sizeof(int)) /
  //         (1024.0 * 1024.0));

  // ========== 第三步：分配并复制到设备 ==========
  cudaError_t err;
  err = cudaMalloc(&d_node_start_, (num_nodes + 1) * sizeof(int));
  if (err != cudaSuccess)
    printf("Error alloc d_node_start_: %s\n", cudaGetErrorString(err));

  err = cudaMalloc(&d_word_index_, total_words_ * sizeof(int));
  if (err != cudaSuccess)
    printf("Error alloc d_word_index_: %s\n", cudaGetErrorString(err));

  err = cudaMalloc(&d_word_data_, total_words_ * sizeof(uint64_t));
  if (err != cudaSuccess)
    printf("Error alloc d_word_data_: %s\n", cudaGetErrorString(err));

  err = cudaMalloc(&d_id_remap_, host_id_remap_.size() * sizeof(int));
  if (err != cudaSuccess)
    printf("Error alloc d_id_remap_: %s\n", cudaGetErrorString(err));

  cudaMemcpy(d_node_start_, host_node_start_.data(),
             (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_word_index_, host_word_index_.data(), total_words_ * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_word_data_, host_word_data_.data(),
             total_words_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_id_remap_, host_id_remap_.data(),
             host_id_remap_.size() * sizeof(int), cudaMemcpyHostToDevice);

  CHECK_CUDA_ERROR();
}

/**
 * 析构函数
 */
__host__ RoaringBitmap::~RoaringBitmap() {
  if (d_node_start_) cudaFree(d_node_start_);
  if (d_word_index_) cudaFree(d_word_index_);
  if (d_word_data_) cudaFree(d_word_data_);
  if (d_id_remap_) cudaFree(d_id_remap_);
}

/**
 * 批量插入 - 空实现，因为我们在主机端构建了位图
 */
__host__ int RoaringBitmap::bulk_insert() { return 0; }
