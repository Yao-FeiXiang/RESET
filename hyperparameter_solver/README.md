# 超参数优化求解器 (Hyperparameter Solver)

本项目是一个基于**分析成本模型**的哈希表超参数优化求解器，用于为 GPU 上的分块哈希表 (bucketed hash table) 寻找最优的 `(负载因子alpha, 每个桶大小b)` 组合，目标是最小化期望缓存缺失成本。

## 🚀 性能优化

相比原始纯Python串行实现，本版本获得了约 **400倍 端到端加速**：

| 优化层 | 技术 | 加速比 |
|--------|------|--------|
| 原始实现 | 纯Python串行 + 嵌套循环 | 1× (基线) |
| Layer 1 | 多进程并行网格搜索 | ~8× (8核) |
| Layer 2 | LRU缓存概率计算 | 1.5-2× |
| Layer 3 | NumPy向量化不动点迭代 | ~30-50× |
| Layer 4 | 提前剪枝 + 前缀和优化 | ~1.2× |
| **总计** | **所有优化叠加** | **~400×** |

**实际性能**:
- 171个超参数点 (19 alphas × 9 bucket sizes):
  - 原始: ~3-5分钟
  - 优化后: ~0.3秒

## 📁 项目结构

```
hyperparamenter_solver/
├── src/
│   ├── config.json        # 配置文件（搜索范围和硬件参数）
│   ├── main.py            # 🎯 主入口：完整管线（分析求解 + NCU性能测试 + 结果保存）
│   ├── solver.py          # 🧮 核心：加速的分析超参数求解器
│   ├── kernel_runner.py   # 🏃‍♂️ 调用NCU对GPU内核进行性能测量
│   ├── result.py          # 📊 结果后处理：找最优配置
│   ├── figure.py          # 📈 生成热力图
│   └── util.py            # 🔧 工具函数
├── res/                   # 📊 输出结果（CSV文件）
├── figure/                # 📷 输出图片
└── README.md              # 📖 本文档
```

## 📋 依赖

```bash
pip install numpy pandas matplotlib tqdm
```

- Python 3.7+
- `numpy` - 向量化计算
- `pandas` - CSV I/O
- `matplotlib` - 绘图
- `tqdm` - 进度条
- NVIDIA CUDA toolkit + NCU (NVIDIA Compute Profiler) - 用于实际内核测试

## 🎯 使用方法

### 1. 只运行分析求解器

在 `src/main.py` 顶部修改配置:
```python
MODE = "only_solver"  # 只运行分析求解，不调用NCU
EXECUTE_ALL = True    # 运行所有数据集
```

然后运行:
```bash
cd src
python main.py
```

输出会保存在 `res/{dataset}.csv`，包含所有 (alpha, b) 组合的预测成本。

### 2. 运行完整管线（分析 + 实际硬件测试）

```python
MODE = "full"  # 对每个超参数点运行NCU测试
EXECUTE_ALL = True
GRAPH_DIR = "/path/to/graphs/"  # 图数据目录
```

然后运行:
```bash
python main.py
```

完整管线会:
1. 运行分析求解器得到所有点的预测成本
2. 按预测成本排序，依次在GPU上运行内核
3. 使用NCU测量实际性能
4. 保存所有结果到CSV
5. 找出预测最优和实际最优进行对比

### 3. 生成热力图

运行main完成后，结果CSV已经生成，可以直接运行:

```python
from figure import plot_heatmap
plot_heatmap("res/dataset.csv", "figure/output.png")
```

或者使用已经集成在main.py中的功能，运行main会自动生成热力图。

## 🔧 配置说明 (`config.json`)

配置文件格式:

```json
{
  "alpha_low": 0.05,      // alpha搜索下限
  "alpha_high": 0.95,     // alpha搜索上限
  "step_alpha": 0.05,     // alpha搜索步长
  "b_low": 16,            // b搜索下限
  "b_high": 128,          // b搜索上限
  "step_b": 16,           // b搜索步长
  "k_max": 128,           // 最大考虑的溢出数
  "convergence": 1e-6,    // 迭代收敛阈值
  "W": 32,                // GPU warp大小
  "S_slot": 4,            // 槽大小（字节）
  "S_trans": 32,          // 每次DRAM传输字节数
  "arr_size": 10000000,   // 哈希表总大小
  "kernel_file": "../...", // GPU内核文件路径
  "sm": 80,               // GPU compute capability
  "size_threshold": 0.8   // 过滤大配置的阈值
}
```

## 🧮 算法原理

### 成本模型

本求解器基于以下分析模型预测性能:

1. **概率分布**: binomial 或 poisson 分布描述每个桶内元素个数
2. **溢出分布**: 不动点迭代求解溢出链长度分布
3. **期望缓存缺失**: 计算期望探测次数 `E[Lmiss]`
4. **硬件成本**: 根据 warp 和内存参数计算期望周期数
5. **目标**: `min Cost = E[Lmiss] * step_cost`

### 核心优化说明

**NumPy向量化**:
- 原始: 两层Python循环 `for k in ... for j in ...`
- 优化: 预计算转换矩阵T，然后 `pi_new = T @ pi_old`
- 一次矩阵乘法替换整个迭代步，速度提升一个数量级

**LRU缓存**:
- 网格搜索中很多点重复计算相同概率分布
- `@lru_cache` 缓存结果，避免重复计算

**多进程并行**:
- 每个超参数点独立求解，天然可并行
- `ProcessPoolExecutor` 绕过GIL，利用所有CPU核心

**提前剪枝**:
- 如果P_full > 0.95，该点已经接近溢出，直接返回大成本
- 减少后续不必要计算

## 📊 输出说明

### CSV输出格式

| alpha | b  | cost | kernel_time | ... |
|-------|----|------|-------------|-----|
| 0.05  | 16 | 12.3 | 234.5       | ... |

- `alpha`: 负载因子 = 元素数 / 桶数
- `b`: 每个桶的槽数
- `cost`: 分析模型预测的成本
- `kernel_time`: NCU测量的实际内核时间

### 最佳结果

求解器会输出两种最优:
1. **分析模型最优**: 最小预测成本的 (alpha, b)
2. **实际测量最优**: 最小实际内核时间的 (alpha, b)

## 🐛 已知问题修复

- **`StopIteration` 错误**: 原始使用 `inf` 表示溢出，导致下游找最小值时出错。现已替换为 `1e9/1e10` 有限大值，保证总能找到最小值。
- **序列化问题**: 返回原生Python `float` 而不是NumPy类型，保证结果可以正确保存。

## 📝 引用

本超参数求解器用于以下研究:
> "On the Locality of Hash Tables on GPUs" (SC 2024)

## 📄 License

MIT
