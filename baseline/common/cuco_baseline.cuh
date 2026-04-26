#ifndef CUCO_BASELINE_CUH
#define CUCO_BASELINE_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_set.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuda/atomic>
#include <limits>
#include <vector>

/**
 * @brief cuCollections静态集合公共基类
 *
 * 封装cuco::static_set的通用操作,减少重复代码
 * 所有基于cuCollections的基线实现都可以继承此类
 *
 * @tparam KeyType 存储的键类型
 */
template <typename KeyType>
class CuCollectionsStaticSetBase {
 public:
  using key_type = KeyType;
  using hasher = cuco::murmurhash3_32<key_type>;
  using probing = cuco::linear_probing<1, hasher>;  // 改用线性探测替代双重哈希
  using set_type = cuco::static_set<key_type, cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<key_type>, probing>;

  /**
   * @brief 构造函数
   * @param capacity 集合的预期容量
   * @param load_factor 负载因子,用于计算实际分配容量,默认为2.0
   */
  explicit CuCollectionsStaticSetBase(std::size_t capacity,
                                      float load_factor = 2.0f)
      : empty_key_(std::numeric_limits<key_type>::max()),
        load_factor_(load_factor),
        set_(nullptr) {
    // 在当前CUDA设备上创建集合
    set_ = new set_type(static_cast<std::size_t>(capacity * load_factor_),
                        cuco::empty_key{empty_key_});
  }

  /**
   * @brief 析构函数
   */
  virtual ~CuCollectionsStaticSetBase() {
    if (set_ != nullptr) {
      delete set_;
    }
  }

  // 禁用拷贝构造和赋值
  CuCollectionsStaticSetBase(const CuCollectionsStaticSetBase&) = delete;
  CuCollectionsStaticSetBase& operator=(const CuCollectionsStaticSetBase&) =
      delete;

  // 禁用移动构造(简化实现)
  CuCollectionsStaticSetBase(CuCollectionsStaticSetBase&&) = delete;

  /**
   * @brief 获取cuCollections集合的contains查询引用,供内核使用
   * @return contains操作的引用
   */
  auto get_contains_ref() const { return set_->ref(cuco::op::contains); }

  /**
   * @brief 插入一批键到集合中
   * @param h_keys 主机端键向量
   * @param stream CUDA流,默认为0
   */
  void insert_keys(thrust::host_vector<key_type>& h_keys,
                   cudaStream_t stream = 0) {
    thrust::device_vector<key_type> d_keys = h_keys;
    set_->insert(d_keys.begin(), d_keys.end(), stream);
    cudaStreamSynchronize(stream);
  }

  /**
   * @brief 兼容旧接口的insert_keys
   * @param h_keys 主机端键向量
   * @param device_id 设备ID(忽略,为了兼容旧代码)
   * @param stream CUDA流,默认为0
   */
  void insert_keys(thrust::host_vector<key_type>& h_keys, int device_id,
                   cudaStream_t stream = 0) {
    insert_keys(h_keys, stream);
  }

 protected:
  key_type empty_key_;  // 空键标记
  set_type* set_;       // cuCollections静态集合指针
  float load_factor_;   // 负载因子,用于计算实际分配容量
};

#endif  // CUCO_BASELINE_CUH
