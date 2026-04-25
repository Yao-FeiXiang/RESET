import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="输入数据集文件夹")
    parser.add_argument("--alpha", type=float, default=0.2, help="哈希表负载因子")
    parser.add_argument("--bucket", type=int, default=5, help="哈希桶大小")
    args = parser.parse_args()

    extra_args = f"--alpha={args.alpha} --bucket={args.bucket}"

    command = f"make && ./ir {args.dataset} {extra_args}"
    os.system(command)

    metrics = [
        "gpu__time_duration.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    ]
    metrics_str = ",".join(metrics)

    command = f"TMPDIR=$HOME/tmp /usr/bin/ncu --metrics {metrics_str} -k ir_kernel ./ir {args.dataset} {extra_args}"
    os.system(command)

if __name__ == "__main__":
    main()