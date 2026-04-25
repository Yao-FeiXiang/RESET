#ifndef GRAPH_DATA_CUH
#define GRAPH_DATA_CUH

#include <string>
#include <vector>

#include "utils.cuh"

// 图/关系数据抽象基类
// 为不同的数据表示提供统一接口：
// - IR(信息检索)使用倒排索引
// - SSS(集合相似性搜索)和 TC(三角形计数)使用CSR图
class GraphData {
 public:
  virtual ~GraphData() = default;

  // 获取节点/词项数量
  virtual int get_num_nodes() const = 0;

  // 获取元素/边总数量
  virtual int get_num_elements() const = 0;

  // 获取主机端偏移向量
  virtual const std::vector<int>& get_host_offsets() const = 0;

  // 获取主机端元素向量
  virtual const std::vector<int>& get_host_elements() const = 0;

  // 获取设备端偏移指针
  virtual int* get_device_offsets() const = 0;

  // 获取设备端元素指针
  virtual int* get_device_elements() const = 0;

 protected:
  GraphData() = default;
};

// 信息检索的倒排索引
// - 词项：行,文档：列
// - 词项 → 包含该词项的文档列表
class InvertedIndex : public GraphData {
 public:
  InvertedIndex(const std::string& index_offsets_path,
                const std::string& index_data_path);

  ~InvertedIndex() override;

  int get_num_nodes() const override { return num_terms_; }
  int get_num_elements() const override { return num_docs_; }
  const std::vector<int>& get_host_offsets() const override {
    return inverted_index_offsets_;
  }
  const std::vector<int>& get_host_elements() const override {
    return inverted_index_;
  }
  int* get_device_offsets() const override { return d_inverted_index_offsets_; }
  int* get_device_elements() const override { return d_inverted_index_; }

 private:
  int num_terms_;
  int num_docs_;
  std::vector<int> inverted_index_offsets_;
  std::vector<int> inverted_index_;

  int* d_inverted_index_offsets_;
  int* d_inverted_index_;
};

// 用于集合相似性搜索和三角形计数的CSR格式图
// - 顶点：行,邻居：列
class CSRGraph : public GraphData {
 public:
  CSRGraph(const std::string& offsets_path, const std::string& columns_path);

  ~CSRGraph() override;

  int get_num_nodes() const override { return num_vertices_; }
  int get_num_elements() const override { return num_edges_; }
  const std::vector<int>& get_host_offsets() const override {
    return csr_offsets_;
  }
  const std::vector<int>& get_host_elements() const override {
    return csr_cols_;
  }
  int* get_device_offsets() const override { return d_csr_offsets_; }
  int* get_device_elements() const override { return d_csr_cols_; }

 private:
  int num_vertices_;
  int num_edges_;
  std::vector<int> csr_offsets_;
  std::vector<int> csr_cols_;

  int* d_csr_offsets_;
  int* d_csr_cols_;
};

#endif  // GRAPH_DATA_CUH
