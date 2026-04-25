import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="输入数据集文件夹")
    parser.add_argument("--alpha", type=float, default=0.25, help="哈希表负载因子")
    parser.add_argument("--bucket", type=int, default=4, help="哈希桶大小")
    parser.add_argument("--total_device", type=int, default=1, help="使用的GPU数量")
    args = parser.parse_args()

    extra_args = f"--alpha={args.alpha} --bucket={args.bucket} --total_device={args.total_device}"

    # command = f"nvcc main_opt.cu --compiler-options -Wall --gpu-architecture=compute_80 -lineinfo --gpu-code=sm_80  -O3 -o main_opt && ./main_opt {dataset}"
    command = f"make && ./ir {args.dataset} {extra_args}"
    os.system(command)

    # command = f"TMPDIR=$HOME/tmp /usr/bin/ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum -k ir_kernel ./ir {dataset}"
    # os.system(command)

if __name__ == "__main__":
    main()