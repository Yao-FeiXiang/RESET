#!/bin/bash
#
# 项目构建脚本
# 功能:
#   1. 搭建Python虚拟环境(.venv)并安装依赖
#   2. 编译CUDA项目
#

set -e  # 遇到错误立即退出

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

echo "========================================"
echo "        项目构建脚本启动"
echo "========================================"
echo ""

# ============================================================================
# 阶段1: Python虚拟环境搭建
# ============================================================================
echo "[1/3] 开始搭建Python虚拟环境..."

if [ -d "$VENV_DIR" ]; then
    echo "  检测到现有虚拟环境，检查完整性..."
    if [ -f "$VENV_DIR/bin/activate" ] && [ -f "$VENV_DIR/bin/python" ]; then
        echo "  虚拟环境已存在，跳过创建，将更新依赖包"
    else
        echo "  虚拟环境损坏，删除重建..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "  创建新的Python虚拟环境..."
    python3 -m venv "$VENV_DIR"
    echo "  虚拟环境创建完成"
fi

echo "  激活虚拟环境..."
source "$VENV_DIR/bin/activate"

echo "  验证Python版本..."
python --version

echo "  升级pip..."
pip install --upgrade pip -q

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "  安装Python依赖包 (requirements.txt)..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "  Python依赖安装完成"
else
    echo "  警告: 未找到requirements.txt文件，跳过依赖安装"
fi

echo ""
echo "[1/3] Python虚拟环境搭建完成 ✓"
echo ""

# ============================================================================
# 阶段2: 编译CUDA项目
# ============================================================================
echo "[2/3] 开始编译CUDA项目..."

# 检查CUDA是否可用
if command -v nvcc &> /dev/null; then
    echo "  检测到CUDA编译器: $(nvcc --version | head -n 1)"
else
    echo "  警告: 未检测到CUDA编译器(nvcc)，跳过CUDA编译"
    echo ""
    echo "========================================"
    echo "        构建脚本完成 (部分跳过)"
    echo "========================================"
    exit 0
fi

# 编译baseline项目
if [ -f "$PROJECT_ROOT/baseline/Makefile" ]; then
    echo "  编译baseline项目..."
    cd "$PROJECT_ROOT/baseline"
    make clean
    make -j$(nproc)
    echo "  baseline项目编译完成"
else
    echo "  未找到baseline/Makefile，跳过baseline编译"
fi

echo ""
echo "[2/3] CUDA项目编译完成 ✓"
echo ""

# ============================================================================
# 阶段3: 环境验证
# ============================================================================
echo "[3/3] 验证构建结果..."

echo "  验证Python环境..."
source "$VENV_DIR/bin/activate"
python -c "import numpy, pandas, matplotlib, seaborn, requests; print('  Python依赖验证通过')"

echo "  验证CUDA可执行文件..."
if [ -f "$PROJECT_ROOT/baseline/sss/sss" ]; then
    echo "    sss 可执行文件: 存在"
else
    echo "    sss 可执行文件: 不存在 (可能编译失败)"
fi

if [ -f "$PROJECT_ROOT/baseline/tc/tc" ]; then
    echo "    tc 可执行文件: 存在"
else
    echo "    tc 可执行文件: 不存在 (可能编译失败)"
fi

if [ -f "$PROJECT_ROOT/baseline/ir/ir" ]; then
    echo "    ir 可执行文件: 存在"
else
    echo "    ir 可执行文件: 不存在 (可能编译失败)"
fi

echo ""
echo "[3/3] 构建结果验证完成 ✓"
echo ""

# ============================================================================
# 总结
# ============================================================================
echo "========================================"
echo "        项目构建全部完成!"
echo "========================================"
echo ""
echo "使用说明:"
echo "  1. 激活虚拟环境: source $VENV_DIR/bin/activate"
echo "  2. 运行预处理: cd preprocess && python run_preprocess.py"
echo "  3. 运行实验: cd baseline/scripts && python run_sss.py"
echo ""
