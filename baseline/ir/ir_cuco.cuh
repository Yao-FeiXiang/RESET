#ifndef IR_CUCO_CUH
#define IR_CUCO_CUH

#include <cuda_runtime.h>

#include <vector>

/**
 * @brief 使用cuCollections基线实现的信息检索交集查询
 *
 * @param inverted_index_num 倒排索引数量
 * @param query_num 查询数量
 * @param d_inverted_index 设备端倒排记录表
 * @param d_inverted_index_offsets 设备端倒排索引偏移
 * @param d_query 设备端查询数组
 * @param d_query_offsets 设备端查询偏移
 * @param d_result 设备端结果数组
 * @param d_result_offsets 设备端结果偏移
 * @param d_result_count 设备端结果计数
 * @param d_G_index 设备端G索引
 * @param CHUNK_SIZE 每个warp处理的查询块大小
 * @param inverted_index_offsets_host 主机端倒排索引偏移
 * @param inverted_index_host 主机端倒排记录
 * @param grid_size 网格大小
 * @param block_size 块大小
 * @param load_factor 负载因子
 * @param stream CUDA流
 * @return int 总结果大小
 */
int run_ir_cuco(int inverted_index_num, int query_num,
                int const* d_inverted_index,
                int const* d_inverted_index_offsets, int const* d_query,
                int const* d_query_offsets, int* d_result,
                long long const* d_result_offsets, int* d_result_count,
                int* d_G_index, int CHUNK_SIZE,
                std::vector<int> const& inverted_index_offsets_host,
                std::vector<int> const& inverted_index_host, int grid_size,
                int block_size, float load_factor = 0.25,
                cudaStream_t stream = 0);

#endif  // IR_CUCO_CUH
