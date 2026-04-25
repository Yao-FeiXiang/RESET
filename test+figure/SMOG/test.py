import os
import subprocess
import re
import threading
import time

def find_graph_folders(root):
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]

def parse_ncu_metrics(output):
    # 匹配 profiler metrics
    # 修改正则以支持 second/msecond/usecond，并分别处理
    
    # 查找 time
    time_pattern = re.compile(r'gpu__time_duration\.sum\s+(second|msecond|usecond)\s+([\d\.]+)')
    time_match = time_pattern.search(output)
    
    # 查找 req
    req_pattern = re.compile(r'l1tex__t_requests_pipe_lsu_mem_global_op_ld\.sum\s+request\s+([\d,]+)')
    req_match = req_pattern.search(output)

    # 查找 sec
    sec_pattern = re.compile(r'l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum\s+sector\s+([\d,]+)')
    sec_match = sec_pattern.search(output)

    if time_match and req_match and sec_match:
        unit = time_match.group(1)
        time_val = float(time_match.group(2))
        
        # 统一转为 ms
        if unit == 'second':
            time_val *= 1000.0
        elif unit == 'usecond':
            time_val /= 1000.0
        
        req = int(req_match.group(1).replace(',', ''))
        sec = int(sec_match.group(1).replace(',', ''))
        return time_val, req, sec
    else:
        return None

def run_benchmark_thread(tasks, script_dir, script_name, N, output_file, mode_name):
    # Resolve absolute path for script directory
    abs_script_dir = os.path.abspath(script_dir)
    
    # Initialize output file with header
    with open(output_file, 'w') as f:
         f.write(f"Dataset\tPattern\tTime(ms)\tReq\tSec\n")
         
    for dataset_name, dataset_path, pattern in tasks:
        print(f"[{mode_name}] Running {dataset_name} {pattern}...")
        
        # Calculate absolute path for dataset to pass to script
        abs_dataset_path = os.path.abspath(dataset_path)
        
        cmd = [
            'python', script_name,
            '--input_graph_folder', abs_dataset_path,
            '--input_pattern', pattern,
            '--N', str(N)
        ]
        
        try:
            # IMPORTANT: Use cwd to run script in its correct directory, 
            # but pass absolute paths for args so they are valid.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=abs_script_dir)
            output = proc.stdout + proc.stderr
            metrics = parse_ncu_metrics(output)
            
            if metrics:
                time_val, req, sec = metrics
                line = f"{dataset_name}\t{pattern}\t{time_val:.3f}\t{req}\t{sec}\n"
                print(f"[{mode_name}] Finished {dataset_name} {pattern}: {time_val:.3f} ms")
            else:
                line = f"{dataset_name}\t{pattern}\tParseError\t0\t0\n"
                print(f"[{mode_name}] Failed parsing {dataset_name} {pattern}")

        except Exception as e:
            line = f"{dataset_name}\t{pattern}\tError: {str(e)}\t0\t0\n"
            print(f"[{mode_name}] Error running {dataset_name} {pattern}: {e}")
            
        with open(output_file, 'a') as f:
            f.write(line)

def main():
    input_graph_root = '../../SMOG/data/processed_graph'
    input_patterns = ['Q0', 'Q1', 'Q2', 'Q3', 'Q5']
    N = 1
    
    # Directories
    normal_dir = '../../SMOG/SMOG_normal'
    hie_dir = '../../SMOG/SMOG_hie'
    
    # Output files
    normal_out = './res_normal.txt'
    hie_out = './res_hie.txt'
    final_out = './res.txt'
    
    folders = find_graph_folders(input_graph_root)
    # Filter folders
    target_folders = []
    for f in folders:
        bname = os.path.basename(f)
        if 'graph500-scale18-ef16' in bname or 'graph500-scale19-ef16' in bname:
            target_folders.append(f)

    # Prepare Tasks
    tasks = []
    for folder in target_folders:
        raw_name = os.path.basename(folder)
        ds_name = 'sc18' if 'scale18' in raw_name else ('sc19' if 'scale19' in raw_name else raw_name)
        for p in input_patterns:
            tasks.append((ds_name, folder, p))

    print(f"Starting benchmark with {len(tasks)} tasks per version...")

    # Create threads
    t1 = threading.Thread(target=run_benchmark_thread, args=(tasks, normal_dir, 'script.py', N, normal_out, 'Normal'))
    t2 = threading.Thread(target=run_benchmark_thread, args=(tasks, hie_dir, 'script_hie.py', N, hie_out, 'Hie'))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
    print("All tests finished. Merging results...")
    
    # Merge results
    normal_data = {} 
    if os.path.exists(normal_out):
        with open(normal_out, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5 and parts[0] != 'Dataset':
                    try:
                        normal_data[(parts[0], parts[1])] = (float(parts[2]), int(parts[3]), int(parts[4]))
                    except:
                        pass

    hie_data = {}
    if os.path.exists(hie_out):
        with open(hie_out, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5 and parts[0] != 'Dataset':
                    try:
                        hie_data[(parts[0], parts[1])] = (float(parts[2]), int(parts[3]), int(parts[4]))
                    except:
                        pass
                        
    # Write combined
    with open(final_out, 'w') as f:
        f.write("Dataset\tPattern\tNormal Time(ms)\tHier Time(ms)\tSpeedup\tNormal Req\tHier Req\tNormal Sec\tHier Sec\tSec/Req(N)\tSec/Req(H)\tSecOpt\n")
        
        for ds, _, p in tasks:
            key = (ds, p)
            if key in normal_data and key in hie_data:
                nt, nr, ns = normal_data[key]
                ht, hr, hs = hie_data[key]
                
                speedup = nt / ht if ht > 0 else 0
                sec_opt = ns / hs if hs > 0 else 0
                sr_n = ns / nr if nr > 0 else 0
                sr_h = hs / hr if hr > 0 else 0
                
                f.write(f"{ds}\t{p}\t{nt:.3f}\t{ht:.3f}\t{speedup:.4f}\t{nr}\t{hr}\t{ns}\t{hs}\t{sr_n:.4f}\t{sr_h:.4f}\t{sec_opt:.4f}\n")
            else:
                nt_str = f"{normal_data[key][0]:.3f}" if key in normal_data else "Missing"
                ht_str = f"{hie_data[key][0]:.3f}" if key in hie_data else "Missing"
                f.write(f"{ds}\t{p}\t{nt_str}\t{ht_str}\t-\t-\t-\t-\t-\t-\t-\t-\n")

    print(f"Results merged into {final_out}")

if __name__ == '__main__':
    main()