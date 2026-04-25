import subprocess
import re

total_devices = [1, 2, 4, 8, 16, 32]

def run_exp(exp_name, exp_dir, dataset_path, output_file, extra_args=None):
    results = {}
    for device in total_devices:
        # 先make
        subprocess.run(["make"], cwd=exp_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # 构造命令
        cmd = [
            f"./{exp_name}",
            dataset_path,
            "--alpha=0.25",
            "--bucket=4",
            f"--total_device={device}"
        ]
        if extra_args:
            cmd += extra_args
        print(f"Running: {' '.join(cmd)} (cwd={exp_dir})")
        proc = subprocess.run(cmd, cwd=exp_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = proc.stdout
        # 提取 Max kernel time
        match = re.search(r"Max kernel time: ([0-9.]+) s", output)
        if match:
            max_time = float(match.group(1))
            results[device] = max_time
        else:
            results[device] = None
        print(f"Device {device}: Max kernel time = {results[device]}")
    # 计算 speedup
    base_time = results.get(1)
    speedups = {}
    if base_time:
        for device, t in results.items():
            if t:
                speedups[device] = base_time / t
            else:
                speedups[device] = None
    else:
        speedups = {device: None for device in total_devices}
    # 写入结果文件
    txt_lines = ["total_device\tmax_kernel_time(s)\tspeedup"]
    for device in total_devices:
        t = results[device]
        s = speedups[device]
        txt_lines.append(f"{device}\t{t if t is not None else 'N/A'}\t{s if s is not None else 'N/A'}")
    with open(output_file, "w") as f:
        f.write("\n".join(txt_lines))
    print(f"Results written to {output_file}")

# tc
run_exp(
    exp_name="tc",
    exp_dir="../../multi-GPU/tc-multi-GPU",
    dataset_path="../../graph_datasets/gplus",
    output_file="./tc_res.txt"
)

# sss
run_exp(
    exp_name="sss",
    exp_dir="../../multi-GPU/sss-multi-GPU",
    dataset_path="../../graph_datasets/gplus",
    output_file="./sss_res.txt"
)

# ir
run_exp(
    exp_name="ir",
    exp_dir="../../multi-GPU/ir-multi-GPU",
    dataset_path="../../ir_datasets/msmarco",
    output_file="./ir_res.txt"
)