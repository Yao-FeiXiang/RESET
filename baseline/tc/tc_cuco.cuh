#ifndef TC_CUCO_CUH
#define TC_CUCO_CUH

#include <cuda_runtime.h>

#include <vector>

/**
 * @brief 使用cuCollections基线实现的三角形计数
 *
 * @param num_nodes 节点数量
 * @param num_edges 边数量
 * @param d_vertexs 设备端顶点数组
 * @param d_csr_row 设备端CSR行数组
 * @param d_csr_cols_for_traversal 设备端遍历用CSR列数组
 * @param csr_row_host 主机端CSR行数组
 * @param csr_cols_host 主机端CSR列数组
 * @param grid_size 网格大小
 * @param block_size 块大小
 * @param CHUNK_SIZE 每个warp处理的查询块大小
 * @param load_factor 负载因子
 * @param stream CUDA流
 * @return unsigned long long 三角形总数
 */
unsigned long long run_tc_cuco(int num_nodes, int num_edges,
                               int const* d_vertexs, int const* d_csr_row,
                               int const* d_csr_cols_for_traversal,
                               std::vector<int> const& csr_row_host,
                               std::vector<int> const& csr_cols_host,
                               int grid_size, int block_size, int CHUNK_SIZE,
                               float load_factor, cudaStream_t stream = 0);

#endif  // TC_CUCO_CUH
