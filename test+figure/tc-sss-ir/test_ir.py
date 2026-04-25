import os
import sys
import subprocess
import re
import argparse
import struct
from tqdm import tqdm

PREPROCESS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../preprocess"))
sys.path.insert(0, PREPROCESS_DIR)

try:
    from main import single_process
    from util import remove_redundance
except ImportError:
    print(f"[Error] Cannot import from {PREPROCESS_DIR}. Please check paths.")
    sys.exit(1)

IR_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../information-retrieval/script.py"))

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

def parse_stats(output_str):
    # 提取你关心的统计项
    stats = {}
    patterns = {
        "inverted_index_offsets size": r"inverted_index_offsets size:\s*([0-9]+)",
        "inverted_index size": r"inverted_index size:\s*([0-9]+)",
        "inverted_index_num": r"inverted_index_num:\s*([0-9]+)",
        "query size": r"query size:\s*([0-9]+)",
        "query_offsets size": r"query_offsets size:\s*([0-9]+)",
        "query_num": r"query_num:\s*([0-9]+)",
        "Max length": r"Max length:\s*([0-9]+)",
        "average degree": r"average degree:\s*([0-9.]+)"
    }
    for key, pat in patterns.items():
        m = re.search(pat, output_str)
        if m:
            stats[key] = m.group(1)
    return stats

def run_benchmark(script_path, dataset_path):
    script_dir = os.path.dirname(script_path)
    cmd = [
        sys.executable, 
        script_path, 
        dataset_path
    ]
    try:
        result = subprocess.run(
            cmd, 
            env=os.environ.copy(), 
            capture_output=True, 
            text=True, 
            cwd=script_dir
        )
        combined_output = result.stdout + result.stderr
        parsed = parse_ncu_output(combined_output)
        stats = parse_stats(combined_output)
        if parsed is None:
            print(f"[Warn] Failed to parse NCU output for {os.path.basename(dataset_path)}")
            return None, combined_output, stats
        return parsed, combined_output, stats
    except Exception as e:
        print(f"[Error] Running benchmark failed: {e}")
        return None, str(e), {}

def save_single_result(filepath, parsed_data, stats):
    labels = ["Baseline (Pre-Opt)", "Optimized (Post-Opt)"]
    with open(filepath, "w") as f:
        # 先写统计信息
        f.write("==== Dataset Stats ====\n")
        for k in [
            "inverted_index_offsets size",
            "inverted_index size",
            "inverted_index_num",
            "query size",
            "query_offsets size",
            "query_num",
            "Max length",
            "average degree"
        ]:
            if k in stats:
                f.write(f"{k}: {stats[k]}\n")
        f.write("\n")
        # 再写ncu指标
        for i, data in enumerate(parsed_data):
            f.write(f"--- {labels[i]} ---\n")
            f.write(f"Time (ms): {data['time']:.4f}\n")
            f.write(f"Sectors:   {data['sectors']:.0f}\n")
            f.write(f"Requests:  {data['requests']:.0f}\n")
            f.write(f"Sec/Req:   {data['sec_per_req']:.4f}\n")
            f.write("\n")

def process_dataset(dataset_name, dataset_path):
    print(f"\n[{dataset_name}] Process start...")
    print(f"[{dataset_name}] Preprocessing (IR)...")
    try:
        single_process(dataset_path, mode="ir", update=True)
    except Exception as e:
        print(f"[Error] Preprocessing failed for {dataset_name}: {e}")
        return None

    print(f"[{dataset_name}] Benchmarking IR...")
    ir_parsed, ir_raw_out, stats = run_benchmark(IR_SCRIPT, dataset_path)
    if ir_parsed:
        save_single_result(os.path.join(dataset_path, "ir_result.txt"), ir_parsed, stats)

    print(f"[{dataset_name}] Cleaning up .bin files...")
    remove_redundance(dataset_path)

    if ir_parsed:
        return {
            "name": dataset_name,
            "ir": ir_parsed,
            "stats": stats
        }
    return None

def main():
    parser = argparse.ArgumentParser(description="Auto Preprocess, Benchmark and Summarize (IR)")
    parser.add_argument("datasets_dir", help="Root folder containing IR dataset subfolders")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.datasets_dir)
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found.")
        sys.exit(1)

    all_results = []
    
    # 收集待处理的数据集列表
    input_datasets = []
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for d in subdirs:
        path = os.path.join(root_dir, d)
        # 取消cqadupstack特殊处理，全部按普通数据集处理
        input_datasets.append((d, path))

    for name, path in tqdm(input_datasets, desc="Processing Datasets"):
        res = process_dataset(name, path)
        if res:
            ir_base = res['ir'][0]['time']
            ir_opt = res['ir'][1]['time']
            speedup = ir_base / ir_opt if ir_opt > 0 else 0
            res['ir_speedup'] = speedup
            all_results.append(res)

    print("\nSorting results and generating summary...")
    all_results.sort(key=lambda x: x['ir_speedup'], reverse=True)
    top_k = all_results

    ir_summary_path = os.path.join(root_dir, "ir_all_res.txt")

    with open(ir_summary_path, "w") as f:
        # 先写表头
        f.write(f"{'Dataset':<25} {'Speedup':<10} {'Time(base/opt)':<25} {'Sectors(base/opt)':<30} {'Seq/Req(base/opt)':<20}\n")
        f.write("-" * 115 + "\n")
        for res in top_k:
            base = res['ir'][0]
            opt = res['ir'][1]
            speedup = res['ir_speedup']
            time_str = f"{base['time']:.2f}/{opt['time']:.2f}"
            sec_str = f"{base['sectors']:.0f}/{opt['sectors']:.0f}"
            spr_str = f"{base['sec_per_req']:.2f}/{opt['sec_per_req']:.2f}"
            f.write(f"{res['name']:<25} {speedup:<10.2f} {time_str:<25} {sec_str:<30} {spr_str:<20}\n")
            # 追加统计信息
            stats = res.get("stats", {})
            for k in [
                "inverted_index_offsets size",
                "inverted_index size",
                "inverted_index_num",
                "query size",
                "query_offsets size",
                "query_num",
                "Max length",
                "average degree"
            ]:
                if k in stats:
                    f.write(f"    {k}: {stats[k]}\n")
            f.write("\n")

    print(f"Done! Summary saved to:\n  {ir_summary_path}")

if __name__ == "__main__":
    main()