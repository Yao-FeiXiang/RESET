#ifndef SSS_BASELINES_H
#define SSS_BASELINES_H

#include <cuda_runtime.h>

#include <utility>
#include <vector>

/**
 * @file sss_baselines.h
 * @brief SSS任务新哈希基线声明
 *
 * 布谷鸟哈希、跳房子哈希、咆哮位图哈希三种基线的运行接口
 */

/**
 * @brief 使用布谷鸟哈希运行集合相似度搜索
 *
 * @param num_pairs 查询对数量
 * @param num_nodes 图节点数量
 * @param d_vertexs 设备端查询顶点对数组
 * @param d_csr_cols_for_vertexs 每个顶点对应的CSR列数组(设备端)
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
std::pair<int, float> run_sss_cuckoo(
    int num_pairs, int num_nodes, int const* d_vertexs,
    int const* d_csr_cols_for_vertexs, int const* d_csr_cols,
    int const* d_csr_offsets, std::vector<int> const& csr_offsets_host,
    std::vector<int> const& csr_cols_host, float threshold, int grid_size,
    int block_size, int CHUNK_SIZE, float load_factor = 0.25,
    cudaStream_t stream = 0);

/**
 * @brief 使用跳房子哈希运行集合相似度搜索
 */
std::pair<int, float> run_sss_hopscotch(
    int num_pairs, int num_nodes, int const* d_vertexs,
    int const* d_csr_cols_for_vertexs, int const* d_csr_cols,
    int const* d_csr_offsets, std::vector<int> const& csr_offsets_host,
    std::vector<int> const& csr_cols_host, float threshold, int grid_size,
    int block_size, int CHUNK_SIZE, float load_factor = 0.25,
    cudaStream_t stream = 0);

/**
 * @brief 使用咆哮位图运行集合相似度搜索
 */
std::pair<int, float> run_sss_roaring(
    int num_pairs, int num_nodes, int const* d_vertexs,
    int const* d_csr_cols_for_vertexs, int const* d_csr_cols,
    int const* d_csr_offsets, std::vector<int> const& csr_offsets_host,
    std::vector<int> const& csr_cols_host, float threshold, int grid_size,
    int block_size, int CHUNK_SIZE, float load_factor = 0.25,
    cudaStream_t stream = 0);

#endif  // SSS_BASELINES_H
