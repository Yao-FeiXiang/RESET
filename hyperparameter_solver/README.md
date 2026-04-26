# 超参数求解器

## 功能概览

### C++ 可调用接口

#### 文件说明
- `src/c_api.py` - Python 侧的 C API 封装
- `cpp/hyper_solver.h` - C++ 头文件
- `cpp/hyper_solver.cpp` - C++ 实现
- `cpp/example_usage.cpp` - 使用示例
- `cpp/Makefile` - 编译脚本示例

#### Python 侧 API

```python
from c_api import solve_hyperparameters, get_cached_hyperparameters

# 求解超参数
alpha, b, cost = solve_hyperparameters(
    arr_size=1000000,        # 哈希表条目数
    W=32,                     # warp大小
    S_trans=128,              # 每次事务字节数
    S_slot=4,                 # 每个槽字节数
    use_cache=True,            # 是否使用缓存
    mode=b'sss'               # 应用模式
)
```

#### C++ 侧 API

```cpp
#include "hyper_solver.h"
using namespace hyper_solver;

// 方式1：直接调用函数
HyperParams result = solve_hyperparameters(1000000);
// result.alpha, result.b, result.cost

// 方式2：使用类接口
HyperSolver solver;
HyperParams result = solver.solve(1000000, true);  // 使用缓存

// 方式3：从数据集路径求解
HyperParams result = solve_hyperparameters_from_dataset(
    "../../graph_datasets/sc18"
);
```

#### 编译 C++ 代码

```bash
cd cpp
make
# 或手动编译
g++ -std=c++17 example_usage.cpp hyper_solver.cpp -o hyper_solver_example
```

---

### 推荐值缓存机制

#### 文件说明
- `src/c_api.py` - 包含 `RecommendationCache` 类
- `src/precompute_cache.py` - 预计算缓存脚本

#### 缓存管理命令

```bash
cd src

# 预计算常见配置的缓存
python3 precompute_cache.py

# 仅显示缓存内容
python3 precompute_cache.py --show

# 清空缓存
python3 precompute_cache.py --clear

# 预计算指定GPU和规模
python3 precompute_cache.py --gpu A100 --size small
```

#### 常见配置说明

缓存覆盖以下组合：

| GPU型号 | 数据规模 | 条目数范围 |
|---------|---------|-----------|
| A100    | small   | 10K-100K |
| V100    | medium  | 500K-2M   |
| 3090    | large   | 5M-20M    |
| 4090    |         |           |
| H100    |         |           |

#### 使用缓存

```python
from c_api import _cache

# 获取缓存
cached = _cache.get(arr_size, W, S_trans, S_slot, mode)
if cached is not None:
    alpha, b, cost = cached

# 存入缓存
_cache.put(arr_size, W, S_trans, S_slot, alpha, b, cost, mode)
```

---

### 结果管理和可视化

#### 文件说明
- `src/result.py` - 结果管理器
- `src/figure.py` - 可视化器
- `src/main.py` - 主程序

#### ResultManager

```python
from result import ResultManager

manager = ResultManager(dataset_name="sc18")

# 获取最佳配置
best = manager.get_best_config(table_type="H")
# best.alpha, best.b, best.cost

# 获取统计信息
stats = manager.get_statistics(table_type="H")
# stats["cost"]["mean"], stats["cost"]["std"], ...

# 导出JSON摘要
summary = manager.export_summary("output_summary.json")
```

#### HeatmapVisualizer

```python
from figure import HeatmapVisualizer, PlotConfig

viz = HeatmapVisualizer(manager, table_type="H")

# 绘制所有指标
viz.plot_all()

# 单独绘制
viz.plot_cost()
viz.plot_kernel_time()
viz.plot_load_sectors()

# 对比多个指标（论文插图专用）
viz.compare_metrics(["cost", "kernel_time"], save_name="comparison")
```

#### 主程序

```bash
cd src

# 仅分析模式（快速）
python3 main.py --mode analyze --datasets sc18 sc19

# 完整模式（含NCU测量）
python3 main.py --mode full --datasets sc18

# 使用缓存
python3 main.py --mode analyze --datasets sc18 --use-cache

# 简化搜索空间（快速测试）
python3 main.py --mode analyze --datasets sc18 --simplify 2

# 预计算缓存
python3 main.py --precompute-cache

# 显示缓存
python3 main.py --show-cache
```

---

## 快速开始

### 1. 安装依赖

```bash
pip3 install numpy pandas matplotlib seaborn tqdm --break-system-packages
```

### 2. 预计算缓存（推荐）

```bash
cd src
python3 precompute_cache.py
```

### 3. 运行分析

```bash
python3 main.py --mode analyze --datasets sc18
```

### 4. 在C++中使用

```cpp
#include "hyper_solver.h"

int main() {
    auto result = hyper_solver::solve_hyperparameters(1000000);
    std::cout << "alpha=" << result.alpha
              << ", b=" << result.b << std::endl;
    return 0;
}
```

---

## 文件结构

```
hyperparameter_solver/
├── src/
│   ├── c_api.py              # C++ API 封装
│   ├── solver.py             # 超参数求解器核心
│   ├── kernel_runner.py      # NCU内核性能分析
│   ├── result.py             # 结果管理器
│   ├── figure.py             # 可视化器
│   ├── main.py               # 主程序
│   ├── precompute_cache.py   # 缓存预计算脚本
│   ├── util.py               # 工具函数
│   └── config.json           # 配置文件
├── cpp/
│   ├── hyper_solver.h        # C++ 头文件
│   ├── hyper_solver.cpp      # C++ 实现
│   ├── example_usage.cpp     # C++ 使用示例
│   └── Makefile              # 编译脚本
├── cache/
│   └── recommended_values.pkl  # 缓存文件
├── res/                      # 结果CSV文件
├── figure/                   # 输出图表
└── README.md                 # 本文档
```

---

## 性能优化说明

### 缓存命中

- 首次计算：根据配置复杂度，从几秒到几分钟
- 缓存命中：几毫秒

### 常见配置

已优化的GPU型号：A100, V100, 3090, 4090, H100
数据规模：small (10K-100K), medium (500K-2M), large (5M-20M)
