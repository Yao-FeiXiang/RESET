import argparse
import os
import struct
import datetime
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from util import *
from merge import *


def build(bin_dir):
    make_dir = Path(bin_dir).parent
    print("[py] Stage0: build binaries (make clean && make)...")
    run_cmd(["make", "clean"], cwd=str(make_dir))
    run_cmd(["make"], cwd=str(make_dir))


def _stage2_worker(bin_build, base_env, part_id, log_path):
    env = dict(base_env)
    env["PART_ID"] = str(part_id)
    run_cmd([bin_build], env=env, log_file=log_path)


def run_stage2_build_partitions(bin_build, base_env, parts, workers, log_path):
    """Stage2: 多进程构建分区 CSR"""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_stage2_worker, bin_build, base_env, p, log_path) for p in range(parts)
        ]

        for f in as_completed(futures):
            f.result()


def run_stage3_merge(out_dir, tmp_dir, parts, n, mode):
    """Stage3: 合并分片（带进度日志）"""

    off_parts = [os.path.join(tmp_dir, f"csr_offsets_part_{p}.bin") for p in range(parts)]
    col_parts = [os.path.join(tmp_dir, f"csr_cols_part_{p}.bin") for p in range(parts)]
    ver_parts = [os.path.join(tmp_dir, f"vertexs_part_{p}.bin") for p in range(parts)]

    offsets_bin = os.path.join(out_dir, "csr_offsets.bin")
    cols_bin = os.path.join(out_dir, "csr_cols.bin")
    vertexs_bin = os.path.join(out_dir, "vertexs.bin")

    check_parts_exist(off_parts, "offsets")
    check_parts_exist(col_parts, "cols")
    print(f"[py][Stage3] all partition files found")

    print(f"[py][Stage3] merging csr_cols ({parts} parts) ...")

    use_vertexs = mode == "sss"
    if use_vertexs:
        print(f"[py][Stage3] mode=sss: checking vertexs parts ...")
        check_parts_exist(ver_parts, "vertexs")
        print(f"[py][Stage3] merging vertexs will be enabled")

    total_edges = merge_partition_outputs(
        cols_bin=cols_bin,
        offsets_bin=offsets_bin,
        col_parts=col_parts,
        off_parts=off_parts,
        n=int(n),
        vertexs_bin=(vertexs_bin if use_vertexs else None),
        ver_parts=(ver_parts if use_vertexs else None),
        log_every=max(1, parts // 10),
    )
    return total_edges


def run():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else Path(__file__).with_name("config.json")
    cfg = resolve_cfg_paths(load_config(cfg_path), cfg_path)
    # ---- config ----
    bcsr_dir = cfg["paths"]["bcsr_dir"]
    work_dir = cfg["paths"]["work_dir"]
    bin_dir = cfg["paths"]["bin_dir"]
    log_dir = cfg["paths"]["log_dir"]

    parts = int(cfg["partition"]["num_parts"])
    mode = cfg["pipeline"]["mode"]
    workers = int(cfg["runtime"]["workers"])
    verbose = bool(cfg["runtime"]["verbose"])

    # ---- dirs ----
    part_dir = os.path.join(work_dir, "parts")
    out_dir = os.path.join(work_dir, "out")
    tmp_dir = os.path.join(work_dir, "tmp")
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("log_dir:", log_dir)
    log_path = os.path.join(log_dir, f"run_{ts}.log")

    # ---- metadata ----
    n = read_u64(os.path.join(bcsr_dir, "num_nodes.bin"))
    if verbose:
        print(f"[py] N={n} parts={parts} mode={mode} workers={workers}")

    bin_extract = os.path.join(bin_dir, "extract_pairs")
    bin_build = os.path.join(bin_dir, "build_csr_partition")
    base_env = {"CONFIG": str(cfg_path.resolve())}

    t_all0 = time.perf_counter()
    try:
        with stage_timer("Build binaries"):
            build(bin_dir)

        # ===== Stage 1 =====
        with stage_timer("Stage1: extract pairs"):
            run_cmd([bin_extract], env=base_env, log_file=log_path)

        # ===== Stage 2 =====
        with stage_timer("Stage2: build csr partitions"):
            run_stage2_build_partitions(bin_build, base_env, parts, workers, log_path)

        # # ===== Stage 3 =====
        with stage_timer("Stage3: merge"):
            total_edges = run_stage3_merge(out_dir, tmp_dir, parts, n, mode)

        # ---- write meta ----
        with stage_timer("Write meta"):
            with open(os.path.join(out_dir, "num_nodes.bin"), "wb") as f:
                f.write(struct.pack("<Q", int(n)))
            with open(os.path.join(out_dir, "num_edges.bin"), "wb") as f:
                f.write(struct.pack("<Q", int(total_edges)))

    finally:
        dt_all = time.perf_counter() - t_all0
        print(f"[py] TOTAL DONE in {dt_all:.3f}s ({fmt_dur(dt_all)})")

    print("[py] DONE")
    print(f"[py] out: {work_dir}")
    print(f"[py] num_edges={total_edges}")


if __name__ == "__main__":
    run()
