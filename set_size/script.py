import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a_size", help="size of set a")
    parser.add_argument("b_size", help="size of set b")
    args = parser.parse_args()


    command = f"make && ./intersection {args.a_size} {args.b_size}"
    os.system(command)

    # metrics = [
    #     "gpu__time_duration.sum",
    #     "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    #     "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    # ]
    # metrics_str = ",".join(metrics)

    # command = f"TMPDIR=$HOME/tmp /usr/bin/ncu --metrics  {metrics_str} -k intersection_kernel ./intersection {args.a_size} {args.b_size}"
    # os.system(command)



if __name__ == "__main__":
    main()