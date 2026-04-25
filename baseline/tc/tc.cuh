#ifndef TC_CUH
#define TC_CUH

#include <cuda_runtime.h>

#include <vector>

#include "../common/graph_data.cuh"
#include "../common/hash_table.cuh"
#include "../common/utils.cuh"

// 三角形计数基线实验
// 统计无向图中三角形的数量
class TCBaseline : public HashTableBuilder {
 public:
  TCBaseline()
      : d_vertexs_(nullptr),
        d_csr_cols_sorted_(nullptr),
        d_total_count_(nullptr),
        d_G_index_(nullptr) {}

  ~TCBaseline();

  // 从CSR准备顶点列表(边源)
  void prepare_vertex_list(const CSRGraph& graph);

  // 使用分层哈希运行实验
  unsigned long long run_hierarchical(CSRGraph& graph, int CHUNK_SIZE,
                                      int grid_size, int block_size,
                                      int bucket_size, bool sorted);

  // 使用普通哈希运行实验
  unsigned long long run_normal(CSRGraph& graph, int CHUNK_SIZE, int grid_size,
                                int block_size, int bucket_size, bool sorted);

  // 获取总三角形数量
  unsigned long long get_total_count();

  // 分配结果缓冲区
  void allocate_buffers();

  // Getter方法供cuco调用
  int* get_d_vertex_list() const { return d_vertexs_; }

 private:
  int num_edges_;

  // 设备端数据
  int* d_vertexs_;
  int* d_csr_cols_sorted_;
  unsigned long long* d_total_count_;
  int* d_G_index_;
};

// TC内核声明
__global__ void tc_kernel(int num_edges, int* vertexs, int* csr_cols,
                          int* csr_offsets, int* hash_length, int* hash_table,
                          long long* hash_table_offsets,
                          unsigned long long* results, int* G_index,
                          int CHUNK_SIZE, bool opt, int max_length,
                          int bucket_num, int bucket_size);

// 辅助函数：在哈希表中搜索(相同实现)
__device__ __forceinline__ bool search_in_hashtable(int key, int* hashtable,
                                                    int bucket_num, int bucket,
                                                    int hash_length,
                                                    int bucket_size);

// 按哈希值排序CSR列,提高局部性
void gpu_sort_csr_cols(std::vector<int>& csr_cols,
                       const std::vector<int>& csr_row, int num_nodes,
                       int max_length);

#endif  // TC_CUH
