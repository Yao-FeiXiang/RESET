import sys
import os
import argparse

USE_NORMAL = True


def main():
    parser = argparse.ArgumentParser()
    dataset = "/data4/cliu26/arabic-2005/out"
    parser.add_argument("--alpha", type=float, default=0.2, help="哈希表负载因子")
    parser.add_argument("--bucket", type=int, default=5, help="哈希桶大小")
    parser.add_argument("--total_device", type=int, default=4, help="设备总数")
    args = parser.parse_args()

    extra_args = f"--alpha={args.alpha} --bucket={args.bucket} --total_device={args.total_device} {'--normal' if USE_NORMAL else ''}"

    metrics = [
        "gpu__time_duration.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    ]
    metrics_str = ",".join(metrics)
    command = f"TMPDIR=$HOME/tmp ncu --metrics  {metrics_str} -k set_similarity_search_kernel ./sss {dataset} {extra_args}"
    os.system(command)


if __name__ == "__main__":
    main()
