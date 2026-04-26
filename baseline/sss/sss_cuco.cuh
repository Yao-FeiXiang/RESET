#ifndef SSS_CUCO_CUH
#define SSS_CUCO_CUH

#include <cuda_runtime.h>

#include <vector>

/**
 * @brief 使用cuCollections基线实现的集合相似性搜索
 *
 * @param num_edges 边数量
 * @param num_nodes 节点数量
 * @param d_vertexs 设备端顶点数组
 * @param d_csr_cols_for_edges 设备边对应的CSR列数组
 * @param d_csr_cols 设备端CSR列数组
 * @param d_csr_offsets 设备端CSR偏移数组
 * @param csr_offsets_host 主机端CSR偏移数组
 * @param csr_cols_host 主机端CSR列数组
 * @param threshold Jaccard相似度阈值
 * @param grid_size 网格大小
 * @param block_size 块大小
 * @param CHUNK_SIZE 每个warp处理的查询块大小
 * @param load_factor 负载因子
 * @param stream CUDA流
 * @return int 符合条件的结果数量
 */
std::pair<int, float> run_sss_cuco(
    int num_edges, int num_nodes, int const* d_vertexs,
    int const* d_csr_cols_for_edges, int const* d_csr_cols,
    int const* d_csr_offsets, std::vector<int> const& csr_offsets_host,
    std::vector<int> const& csr_cols_host, float threshold, int grid_size,
    int block_size, int CHUNK_SIZE, float load_factor = 0.25,
    cudaStream_t stream = 0);

#endif  // SSS_CUCO_CUH
