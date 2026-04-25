import sys
import os
import argparse


def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    dataset = sys.argv[1]

    # command = f"nvcc main_opt.cu --compiler-options -Wall --gpu-architecture=compute_80 -lineinfo --gpu-code=sm_80  -O3 -o main_opt && ./main_opt {dataset}"
    command = f"make && ./ir {dataset}"
    os.system(command)

    command = f"TMPDIR=$HOME/tmp /usr/bin/ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum -k ir_kernel ./ir {dataset}"
    os.system(command)


if __name__ == "__main__":
    main()
