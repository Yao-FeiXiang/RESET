import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="输入数据集文件夹")
    parser.add_argument("--alpha", type=float, default=0.25, help="哈希表负载因子")
    parser.add_argument("--bucket", type=int, default=4, help="哈希桶大小")
    parser.add_argument("--total_device", type=int, default=1, help="设备总数")
    args = parser.parse_args()

    extra_args = f"--alpha={args.alpha} --bucket={args.bucket} --total_device={args.total_device}"

    command = f"make && ./tc {args.dataset} {extra_args}"
    os.system(command)


if __name__ == "__main__":
    main()