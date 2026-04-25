import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser()

    dataset = "../graph_datasets/bio-mouse-gene"
    parser.add_argument("--alpha", type=float, default=0.2, help="哈希表负载因子")
    parser.add_argument("--bucket", type=int, default=5, help="哈希桶大小")
    args = parser.parse_args()

    extra_args = f"--alpha={args.alpha} --bucket={args.bucket}"

    command = f"make && ./tc {dataset} {extra_args}"
    os.system(command)

    metrics = [
        "gpu__time_duration.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    ]
    metrics_str = ",".join(metrics)

    command = f"TMPDIR=$HOME/tmp ncu --metrics  {metrics_str} -k dynamic_triangle_count ./tc {dataset} {extra_args}"
    os.system(command)


if __name__ == "__main__":
    main()
