#ifndef IR_CUH
#define IR_CUH

#include <utility>
#include <vector>

#include "../common/graph_data.cuh"
#include "../common/hash_table.cuh"
#include "../common/utils.cuh"

// 信息检索基线实验
// 对多个文档链表求交集
class IRBaseline : public HashTableBuilder {
 public:
  IRBaseline()
      : d_inverted_index_(nullptr),
        d_inverted_index_offsets_(nullptr),
        d_inverted_index_sorted_(nullptr),
        d_query_(nullptr),
        d_query_offsets_(nullptr),
        d_result_(nullptr),
        d_result_offsets_(nullptr),
        d_result_count_(nullptr),
        d_G_index_(nullptr) {}

  ~IRBaseline();

  // 预排序倒排索引(用于hierarchical哈希),在计时前完成
  // 使用哈希表直接提取，无需排序
  void pre_sort_inverted_index(const InvertedIndex& index,
                               int slots_per_bucket);

  // 从文件加载查询数据
  void load_queries(const std::string& query_path,
                    const std::string& query_offsets_path,
                    const std::string& query_num_path);

  // 使用分层哈希运行实验
  std::pair<int, float> run_hierarchical(int CHUNK_SIZE, int grid_size,
                                         int block_size, int bucket_size,
                                         bool sorted);

  // 使用普通哈希运行实验
  std::pair<int, float> run_normal(int CHUNK_SIZE, int grid_size,
                                   int block_size, int bucket_size,
                                   bool sorted);

  // 获取所有查询结果计数
  std::vector<int> get_results();

  // 分配结果缓冲区
  void allocate_result_buffers(const InvertedIndex& index);

  // Getter方法供cuco调用
  int get_query_num() const { return query_num_; }
  int const* get_d_inverted_index() const { return d_inverted_index_; }
  int const* get_d_inverted_index_offsets() const {
    return d_inverted_index_offsets_;
  }
  int const* get_d_query() const { return d_query_; }
  int const* get_d_query_offsets() const { return d_query_offsets_; }
  int* get_d_result() const { return d_result_; }
  long long const* get_d_result_offsets() const { return d_result_offsets_; }
  int* get_d_result_count() const { return d_result_count_; }
  int* get_d_G_index() const { return d_G_index_; }

 private:
  // 设备端倒排索引数据
  int* d_inverted_index_;
  int* d_inverted_index_offsets_;
  int* d_inverted_index_sorted_;  // 按桶位置排序后的倒排索引

  // 主机端查询数据
  std::vector<int> global_query_;
  std::vector<int> query_offsets_;
  int query_num_;

  // 设备端查询数据
  int* d_query_;
  int* d_query_offsets_;

  // 设备端结果数据
  int* d_result_;
  long long* d_result_offsets_;
  int* d_result_count_;
  int* d_G_index_;
};

// IR搜索内核声明
__global__ void ir_kernel(int* inverted_index, int* inverted_index_offsets,
                          int* query, int* query_offsets, int query_num,
                          int* result, long long* result_offsets,
                          int* result_count, int* G_index, int CHUNK_SIZE,
                          int max_length, bool opt, int* hashtable,
                          long long* hashtable_offset, int bucket_num,
                          int bucket_size);

// 辅助函数：在哈希表中搜索
__device__ __forceinline__ bool search_in_hashtable(int key, int* hashtable,
                                                    int bucket_num, int bucket,
                                                    int hash_length,
                                                    int bucket_size);

#endif  // IR_CUH
