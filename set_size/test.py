import subprocess
import re
import numpy as np

sizes = [2**i for i in range(6, 16)]  # 64, 128, ..., 32768

result_lines = []
result_lines.append("a_size\tb_size\tNormalTime(ms)\tOptTime(ms)\tTimeSpeedup\tNormalSectors\tOptSectors\tMemSpeedup\n")

repeat = 20

for a in sizes:
    for b in sizes:
        print(f"Testing a={a}, b={b} ...")
        normal_times = []
        opt_times = []
        normal_sectors = []
        opt_sectors = []

        for _ in range(repeat):
            proc = subprocess.run(
                ["python3", "script.py", str(a), str(b)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            output = proc.stdout

            # 匹配 Normal/Optimized Intersection Time
            normal_time_match = re.search(r"Normal Intersection Time:\s*([\d.]+)\s*ms", output)
            opt_time_match = re.search(r"Optimized Intersection Time:\s*([\d.]+)\s*ms", output)
            if normal_time_match:
                normal_times.append(float(normal_time_match.group(1)))
            if opt_time_match:
                opt_times.append(float(opt_time_match.group(1)))

            # 匹配 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
            sectors = re.findall(r"l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum\s+\w+\s+([\d,]+)", output)
            if len(sectors) >= 2:
                normal_sectors.append(int(sectors[0].replace(",", "")))
                opt_sectors.append(int(sectors[1].replace(",", "")))

        # 取平均
        normal_time_avg = np.mean(normal_times) if normal_times else None
        opt_time_avg = np.mean(opt_times) if opt_times else None
        normal_sectors_avg = np.mean(normal_sectors) if normal_sectors else None
        opt_sectors_avg = np.mean(opt_sectors) if opt_sectors else None

        # 计算加速比
        time_speedup = round(normal_time_avg / opt_time_avg, 3) if normal_time_avg and opt_time_avg else "N/A"
        mem_speedup = round(normal_sectors_avg / opt_sectors_avg, 3) if normal_sectors_avg and opt_sectors_avg else "N/A"

        result_lines.append(f"{a}\t{b}\t{normal_time_avg}\t{opt_time_avg}\t{time_speedup}\t{normal_sectors_avg}\t{opt_sectors_avg}\t{mem_speedup}\n")

# 写入结果文件
with open("new_result.txt", "w") as f:
    f.writelines(result_lines)

print("测试完成，结果已写入 result.txt")