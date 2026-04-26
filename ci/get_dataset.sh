#!/bin/bash
# 一键执行下载和预处理任务
set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Starting data pipeline..."
echo "========================================"

# 激活虚拟环境
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Warning: .venv not found, using system Python..."
fi

# 步骤1: 下载数据集
echo ""
echo "========================================"
echo "Step 1: Downloading datasets..."
echo "========================================"
cd "$PROJECT_ROOT/preprocess"
python download.py
if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
fi

# 步骤2: 执行预处理
echo ""
echo "========================================"
echo "Step 2: Running preprocessing..."
echo "========================================"

# 处理所有模式
for mode in "sss" "tc" "ir"; do
    echo ""
    echo "Processing mode: $mode..."
    python run_preprocess.py -m "$mode"
    if [ $? -ne 0 ]; then
        echo "Preprocessing for $mode failed!"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All tasks completed successfully!"
echo "========================================"
