#ifndef HASH_TABLE_CUH
#define HASH_TABLE_CUH

#include <cub/cub.cuh>
#include <vector>

#include "utils.cuh"

// 哈希表构建器基类
// 提供分层哈希和普通哈希两种哈希表的通用构建逻辑
// 所有实验共享此基类,消除代码重复
class HashTableBuilder {
 public:
  HashTableBuilder()
      : d_hash_table_hierarchical_(nullptr),
        d_hash_table_normal_(nullptr),
        d_hash_length_(nullptr),
        d_hash_tables_offset_(nullptr),
        hash_table_hierarchical_(nullptr),
        hash_table_normal_(nullptr) {}

  virtual ~HashTableBuilder() {
    // 清理设备端内存
    if (d_hash_table_hierarchical_) cudaFree(d_hash_table_hierarchical_);
    if (d_hash_table_normal_) cudaFree(d_hash_table_normal_);
    if (d_hash_length_) cudaFree(d_hash_length_);
    if (d_hash_tables_offset_) cudaFree(d_hash_tables_offset_);

    // 清理主机端内存
    if (hash_table_hierarchical_) delete[] hash_table_hierarchical_;
    if (hash_table_normal_) delete[] hash_table_normal_;
  }

  // 禁用拷贝构造和赋值,避免double free
  HashTableBuilder(const HashTableBuilder&) = delete;
  HashTableBuilder& operator=(const HashTableBuilder&) = delete;

  // 构建分层哈希和普通哈希两张哈希表
  void build_hash_tables(int num_nodes, const std::vector<int>& h_offsets,
                         const std::vector<int>& h_elements, float load_factor,
                         int bucket_size, const std::vector<int>& vertexs = {},
                         const std::vector<int>& vertex_csr_cols = {});

  // 获取设备端指针的getter方法(供内核使用)
  int* get_d_hash_hierarchical() const { return d_hash_table_hierarchical_; }
  int* get_d_hash_normal() const { return d_hash_table_normal_; }
  int* get_d_hash_length() const { return d_hash_length_; }
  long long* get_d_hash_tables_offset() const { return d_hash_tables_offset_; }
  int get_max_length() const { return max_length_; }
  long long get_bucket_num() const { return bucket_num_; }

 protected:
  // 设备端指针 - 对子类可见,用于启动内核
  int* d_hash_table_hierarchical_;
  int* d_hash_table_normal_;
  int* d_hash_length_;
  long long* d_hash_tables_offset_;

  // 主机端指针
  int* hash_table_hierarchical_;
  int* hash_table_normal_;

  int max_length_;        // 最大哈希长度
  long long bucket_num_;  // 总桶数量
};

// 辅助内核 - 这些内核所有实验共享
__global__ void calculate_hash_length_kernel(int* hash_length, int num_nodes,
                                             int* offsets, int* max_length,
                                             float load_factor,
                                             int bucket_size);

// 构建分层哈希表内核
__global__ void build_hierarchical_kernel(int* hash_table, int* hash_length,
                                          long long* hash_tables_offset,
                                          int element_count, int bucket_num,
                                          int max_length, int* node_ids,
                                          int* elements, int* conflict_count,
                                          int bucket_size);

// 构建普通哈希表内核
__global__ void build_normal_kernel(int* hash_table, int* hash_length,
                                    long long* hash_tables_offset,
                                    int element_count, int bucket_num,
                                    int* node_ids, int* elements,
                                    int* conflict_count, int bucket_size);

#endif  // HASH_TABLE_CUH
