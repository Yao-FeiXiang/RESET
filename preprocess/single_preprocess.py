from main import single_process

# 指定你的数据集文件夹路径和模式
dataset_path = "../ir_datasets/lotte"  # 修改为你的实际路径
mode = "ir"  # 或 "tc" 或 "ir"

single_process(dataset_path, mode, update=True)