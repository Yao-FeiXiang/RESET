import os
import sys
import subprocess
import re
import argparse
import struct
from tqdm import tqdm

# 添加 preprocess 路径以便调用
PREPROCESS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../preprocess"))
sys.path.insert(0, PREPROCESS_DIR)

try:
    from main import single_process
    from util import remove_redundance
except ImportError:
    print(f"[Error] Cannot import from {PREPROCESS_DIR}. Please check paths.")
    sys.exit(1)

TC_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../triangle-count/script.py"))
SSS_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../set-similarity-search/script.py"))

def parse_ncu_output(output_str):
    regex_time = r"gpu__time_duration\.sum\s+\w+\s+([\d,.]+)"
    regex_req = r"l1tex__t_requests_pipe_lsu_mem_global_op_ld\.sum\s+\w+\s+([\d,.]+)"
    regex_sec = r"l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum\s+\w+\s+([\d,.]+)"

    def clean_num(s):
        return float(s.replace(",", ""))

    times = [clean_num(x) for x in re.findall(regex_time, output_str)]
    reqs = [clean_num(x) for x in re.findall(regex_req, output_str)]
    secs = [clean_num(x) for x in re.findall(regex_sec, output_str)]

    results = []
    if len(times) >= 2 and len(reqs) >= 2 and len(secs) >= 2:
        for i in range(2):
            t = times[i]
            r = reqs[i]
            s = secs[i]
            ratio = s / r if r > 0 else 0
            results.append({
                "time": t,
                "sectors": s,
                "requests": r,
                "sec_per_req": ratio
            })
        return results 
    return None

def run_benchmark(script_path, dataset_path, alpha, bucket):
    # 获取 script.py 所在的文件夹路径，作为工作目录
    script_dir = os.path.dirname(script_path)
    
    cmd = [
        sys.executable, 
        script_path, 
        dataset_path, 
        f"--alpha={alpha}", 
        f"--bucket={bucket}"
    ]
    
    try:
        # 关键修复: cwd=script_dir 确保在正确目录下执行 make/exec
        result = subprocess.run(
            cmd, 
            env=os.environ.copy(), 
            capture_output=True, 
            text=True, 
            cwd=script_dir
        )
        combined_output = result.stdout + result.stderr
        parsed = parse_ncu_output(combined_output)
        
        if parsed is None:
            print(f"[Warn] Failed to parse NCU output for {os.path.basename(dataset_path)}")
            # print(combined_output) # Debug if needed
            return None, combined_output
            
        return parsed, combined_output
    except Exception as e:
        print(f"[Error] Running benchmark failed: {e}")
        return None, str(e)

def save_single_result(filepath, parsed_data, alpha, bucket):
    labels = ["Baseline (Pre-Opt)", "Optimized (Post-Opt)"]
    with open(filepath, "w") as f:
        f.write(f"Parameters: alpha={alpha}, bucket={bucket}\n\n")
        for i, data in enumerate(parsed_data):
            f.write(f"--- {labels[i]} ---\n")
            f.write(f"Time (ms): {data['time']:.4f}\n")
            f.write(f"Sectors:   {data['sectors']:.0f}\n")
            f.write(f"Requests:  {data['requests']:.0f}\n")
            f.write(f"Sec/Req:   {data['sec_per_req']:.4f}\n")
            f.write("\n")

def read_hyperparameters(dataset_path):
    alpha = 0.25 
    bucket = 4   
    
    alpha_path = os.path.join(dataset_path, "load_factor.bin")
    bucket_path = os.path.join(dataset_path, "bucket_size.bin")

    try:
        if os.path.exists(alpha_path):
            with open(alpha_path, "rb") as f:
                content = f.read(4)
                if len(content) == 4:
                    alpha = struct.unpack("f", content)[0]
        
        if os.path.exists(bucket_path):
            with open(bucket_path, "rb") as f:
                content = f.read(4)
                if len(content) == 4:
                    bucket = struct.unpack("i", content)[0]
        
        print(f"Loaded params for {os.path.basename(dataset_path)}: alpha={alpha:.2f}, bucket={bucket}")
    except Exception as e:
        print(f"[Warn] Could not read params, using defaults. Error: {e}")

    return alpha, bucket

def process_dataset(dataset_name, dataset_path):
    print(f"\n[{dataset_name}] Process start...")
    
    print(f"[{dataset_name}] Preprocessing (SSS + HyperParams)...")
    try:
        single_process(dataset_path, mode="sss", update=True, calculate_hyperparameters=True)
        single_process(dataset_path, mode="tc", update=True, calculate_hyperparameters=False)
    except Exception as e:
        print(f"[Error] Preprocessing failed for {dataset_name}: {e}")
        return None

    alpha, bucket = read_hyperparameters(dataset_path)

    print(f"[{dataset_name}] Benchmarking TC (alpha={alpha:.2f}, b={bucket})...")
    tc_parsed, tc_raw_out = run_benchmark(TC_SCRIPT, dataset_path, alpha, bucket)
    if tc_parsed:
        save_single_result(os.path.join(dataset_path, "tc_result.txt"), tc_parsed, alpha, bucket)
    
    print(f"[{dataset_name}] Benchmarking SSS (alpha={alpha:.2f}, b={bucket})...")
    sss_parsed, sss_raw_out = run_benchmark(SSS_SCRIPT, dataset_path, alpha, bucket)
    if sss_parsed:
        save_single_result(os.path.join(dataset_path, "sss_result.txt"), sss_parsed, alpha, bucket)

    print(f"[{dataset_name}] Cleaning up .bin files...")
    remove_redundance(dataset_path)

    if tc_parsed and sss_parsed:
        return {
            "name": dataset_name,
            "tc": tc_parsed,   
            "sss": sss_parsed,
            "alpha": alpha, 
            "bucket": bucket
        }
    return None

def main():
    parser = argparse.ArgumentParser(description="Auto Preprocess, Benchmark and Summarize")
    parser.add_argument("datasets_dir", help="Root folder containing dataset subfolders")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.datasets_dir)
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found.")
        sys.exit(1)

    all_results = []
    
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    for d in tqdm(subdirs, desc="Processing Datasets"):
        path = os.path.join(root_dir, d)
        res = process_dataset(d, path)
        if res:
            tc_base = res['tc'][0]['time']
            tc_opt = res['tc'][1]['time']
            speedup = tc_base / tc_opt if tc_opt > 0 else 0
            res['tc_speedup'] = speedup
            all_results.append(res)

    print("\nSorting results and generating summary...")
    all_results.sort(key=lambda x: x['tc_speedup'], reverse=True)
    top_k = all_results
    
    tc_summary_path = os.path.join(root_dir, "tc_all_res.txt")
    sss_summary_path = os.path.join(root_dir, "sss_all_res.txt")

    # 写入 TC 汇总
    with open(tc_summary_path, "w") as f:
        # 表头
        f.write(f"{'Dataset':<25} {'Speedup':<10} {'Time(base/opt)':<25} {'Sectors(base/opt)':<30} {'Seq/Req(base/opt)':<20}\n")
        f.write("-" * 115 + "\n")
        for res in top_k:
            base = res['tc'][0]
            opt = res['tc'][1]
            speedup = res['tc_speedup']
            
            time_str = f"{base['time']:.2f}/{opt['time']:.2f}"
            sec_str = f"{base['sectors']:.0f}/{opt['sectors']:.0f}"
            spr_str = f"{base['sec_per_req']:.2f}/{opt['sec_per_req']:.2f}"

            f.write(f"{res['name']:<25} {speedup:<10.2f} {time_str:<25} {sec_str:<30} {spr_str:<20}\n")

    # 写入 SSS 汇总
    with open(sss_summary_path, "w") as f:
        # 表头
        f.write(f"{'Dataset':<25} {'Speedup':<10} {'Time(base/opt)':<25} {'Sectors(base/opt)':<30} {'Seq/Req(base/opt)':<20}\n")
        f.write("-" * 115 + "\n")
        for res in top_k:
            base = res['sss'][0]
            opt = res['sss'][1]
            # SSS 计算加速比
            sss_speedup = base['time'] / opt['time'] if opt['time'] > 0 else 0
            
            time_str = f"{base['time']:.2f}/{opt['time']:.2f}"
            sec_str = f"{base['sectors']:.0f}/{opt['sectors']:.0f}"
            spr_str = f"{base['sec_per_req']:.2f}/{opt['sec_per_req']:.2f}"

            f.write(f"{res['name']:<25} {sss_speedup:<10.2f} {time_str:<25} {sec_str:<30} {spr_str:<20}\n")

    print(f"Done! Summaries saved to:\n  {tc_summary_path}\n  {sss_summary_path}")

if __name__ == "__main__":
    main()