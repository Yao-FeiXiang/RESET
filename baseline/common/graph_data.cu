#include "graph_data.cuh"

InvertedIndex::InvertedIndex(const std::string& index_offsets_path,
                             const std::string& index_data_path) {
  read_i32_vec(index_offsets_path, inverted_index_offsets_);
  read_i32_vec(index_data_path, inverted_index_);

  num_terms_ = inverted_index_offsets_.size() - 1;
  num_docs_ = inverted_index_.size();

  // 分配设备内存
  cudaMalloc(&d_inverted_index_offsets_, sizeof(int) * (num_terms_ + 1));
  cudaMalloc(&d_inverted_index_, sizeof(int) * num_docs_);

  // 拷贝数据到设备
  cudaMemcpy(d_inverted_index_offsets_, inverted_index_offsets_.data(),
             sizeof(int) * (num_terms_ + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_inverted_index_, inverted_index_.data(), sizeof(int) * num_docs_,
             cudaMemcpyHostToDevice);
}

InvertedIndex::~InvertedIndex() {
  if (d_inverted_index_offsets_) cudaFree(d_inverted_index_offsets_);
  if (d_inverted_index_) cudaFree(d_inverted_index_);
}

// CSR图实现
CSRGraph::CSRGraph(const std::string& offsets_path,
                   const std::string& columns_path) {
  read_i32_vec(offsets_path, csr_offsets_);
  read_i32_vec(columns_path, csr_cols_);

  num_vertices_ = csr_offsets_.size() - 1;
  num_edges_ = csr_cols_.size();

  // 分配设备内存
  cudaMalloc(&d_csr_offsets_, sizeof(int) * (num_vertices_ + 1));
  cudaMalloc(&d_csr_cols_, sizeof(int) * num_edges_);

  // 拷贝数据到设备
  cudaMemcpy(d_csr_offsets_, csr_offsets_.data(),
             sizeof(int) * (num_vertices_ + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_cols_, csr_cols_.data(), sizeof(int) * num_edges_,
             cudaMemcpyHostToDevice);
}

CSRGraph::~CSRGraph() {
  if (d_csr_offsets_) cudaFree(d_csr_offsets_);
  if (d_csr_cols_) cudaFree(d_csr_cols_);
}
