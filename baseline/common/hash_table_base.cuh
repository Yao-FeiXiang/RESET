#ifndef HASH_TABLE_BASE_CUH
#define HASH_TABLE_BASE_CUH

/**
 * @file hash_table_base.cuh
 * @brief 哈希表抽象基类,定义统一接口规范
 *
 * 所有哈希表实现都必须继承此类,实现纯虚函数接口
 * 保证不同哈希表实现可以在main函数中通过命令行参数选择调用
 */

#include <cuda_runtime.h>

/**
 * @brief 哈希表抽象基类
 * @tparam KeyType 键的类型
 *
 * 定义了哈希表必须实现的基本操作接口：
 * - 单个元素操作：insert/contains/remove
 * - 批量操作：bulk_insert/bulk_contains
 * - 容量信息查询
 */
template <typename KeyType>
class HashTableBase {
 public:
  /**
   * @brief 虚析构函数,保证正确析构子类对象
   */
  virtual ~HashTableBase() = default;

  /**
   * @brief 插入单个键(仅主机端)
   * @param key 要插入的键
   * @return 插入是否成功(true表示成功,false表示失败)
   */
  __host__ virtual bool insert(KeyType key) = 0;

  /**
   * @brief 查询单个键是否存在(可主机端或设备端调用)
   * @param key 要查询的键
   * @return true表示存在,false表示不存在
   */
  __host__ __device__ virtual bool contains(KeyType key) const = 0;

  /**
   * @brief 删除单个键(仅主机端)
   * @param key 要删除的键
   * @return 删除是否成功(true表示成功删除,false表示键不存在)
   */
  __host__ virtual bool remove(KeyType key) = 0;

  /**
   * @brief 批量插入多个键
   * @param keys 主机端键数组指针
   * @param n 插入数量
   * @note 输入keys在主机端,方法内部需要负责拷贝到设备并插入
   */
  __host__ virtual void bulk_insert(const KeyType* keys, size_t n) = 0;

  /**
   * @brief 批量查询多个键是否存在
   * @param keys 主机端要查询的键数组
   * @param results 主机端结果数组,true表示存在,false表示不存在
   * @param n 查询数量
   * @note 输入keys在主机端,输出results也在主机端
   */
  __host__ virtual void bulk_contains(const KeyType* keys, bool* results,
                                      size_t n) const = 0;

  /**
   * @brief 获取哈希表容量(总槽位数)
   * @return 哈希表总容量
   */
  __host__ __device__ virtual size_t capacity() const = 0;

  /**
   * @brief 获取已插入元素数量
   * @return 已插入元素数量
   */
  __host__ __device__ virtual size_t size() const = 0;

  /**
   * @brief 获取当前负载因子
   * @return size() / capacity()
   */
  float load_factor() const {
    return static_cast<float>(size()) / static_cast<float>(capacity());
  }
};

#endif  // HASH_TABLE_BASE_CUH
