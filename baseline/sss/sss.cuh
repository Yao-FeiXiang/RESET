#ifndef SSS_CUH
#define SSS_CUH

#include <vector>

#include "../common/graph_data.cuh"
#include "../common/hash_table.cuh"
#include "../common/utils.cuh"

// 集合相似性搜索基线实验
// 计算顶点邻居集合之间的Jaccard相似度
class SSSBaseline : public HashTableBuilder {
 public:
  SSSBaseline()
      : d_vertexs_(nullptr),
        d_csr_cols_sorted_(nullptr),
        d_results_(nullptr),
        d_G_index_(nullptr) {}

  ~SSSBaseline();

  // 从文件加载顶点对
  void load_vertex_pairs(const std::string& vertexs_path);

  // 使用分层哈希运行实验,返回(结果数, 内核执行时间ms)
  std::pair<int, float> run_hierarchical(CSRGraph& graph, int CHUNK_SIZE,
                                         int grid_size, int block_size,
                                         int bucket_size, float threshold,
                                         bool sorted);

  // 使用普通哈希运行实验,返回(结果数, 内核执行时间ms)
  std::pair<int, float> run_normal(CSRGraph& graph, int CHUNK_SIZE,
                                   int grid_size, int block_size,
                                   int bucket_size, float threshold,
                                   bool sorted);

  // 获取超过阈值的对数
  int get_result_count();

  // 分配结果缓冲区
  void allocate_buffers();

  // 优化版：从分层哈希表直接提取，按哈希桶顺序重排CSR列（计时外预处理）
  void reorder_csr_by_hash_layout(CSRGraph& graph, int slots_per_bucket);

  // 兼容旧接口（内部调用新接口）
  void pre_sort_csr_cols(CSRGraph& graph, int bucket_size) {
    reorder_csr_by_hash_layout(graph, bucket_size);
  }

  // 获取排序后的CSR列指针
  int* get_sorted_csr_cols() const { return d_csr_cols_sorted_; }

  // Getter方法供cuco调用
  int get_num_pairs() const { return num_edges_; }
  int* get_d_vertexs() const { return d_vertexs_; }
  int* get_d_csr_cols_for_vertexs() const { return d_csr_cols_sorted_; }

 private:
  // 主机端顶点对
  std::vector<int> vertexs_;
  int num_edges_;

  // 设备端数据
  int* d_vertexs_;
  int* d_csr_cols_sorted_;
  int* d_results_;
  int* d_G_index_;
};

// SSS内核声明
__global__ void sss_kernel(int num_edges, int const* __restrict__ vertexs,
                           int* csr_cols, int const* __restrict__ csr_offsets,
                           int const* __restrict__ hash_length, int* hash_table,
                           long long* hash_table_offsets, int* results,
                           int* G_index, int CHUNK_SIZE, bool opt,
                           int max_length, int bucket_num, float threshold,
                           int bucket_size);

// 辅助函数：在哈希表中搜索(与IR实现相同)
__device__ __forceinline__ bool search_in_hashtable(int key, int* hashtable,
                                                    int bucket_num, int bucket,
                                                    int hash_length,
                                                    int bucket_size);

#endif  // SSS_CUH
