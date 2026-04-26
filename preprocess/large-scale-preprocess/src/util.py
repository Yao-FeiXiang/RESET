import subprocess
from typing import List, Optional, Any, Dict
import json
from pathlib import Path
import os
import struct
import time
import contextlib


def read_u64(path: str) -> int:
    """读取 8 字节小端 u64 元数据。"""
    with open(path, "rb") as f:
        b = f.read(8)
    if len(b) != 8:
        raise RuntimeError(f"bad u64 file: {path}")

    return struct.unpack("<Q", b)[0]


import subprocess
import sys


def run_cmd(cmd, env=None, cwd=None, log_file=None):
    """
    运行外部命令：
    - stdout / stderr 同时写到终端和 log_file
    """
    if log_file is None:
        p = subprocess.run(cmd, env=env, cwd=cwd)
        if p.returncode != 0:
            raise RuntimeError(f"Command failed (rc={p.returncode}): {' '.join(cmd)}")
        return

    with open(log_file, "a") as lf:
        lf.write(f"\n===== RUN: {' '.join(cmd)} =====\n")
        lf.flush()
        p = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = p.wait()
        if rc != 0:
            lf.write(f"[py][ERR] command failed rc={rc}\n")
            raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict[str, Any], config_path: str | Path) -> None:
    config_path = Path(config_path)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def resolve_cfg_paths(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """
    将 cfg["paths"] 中的所有路径统一解析为绝对路径。
    """
    base = cfg_path.parent.resolve()

    def _abs(p: str) -> str:
        pp = Path(p)
        return str(pp.resolve()) if pp.is_absolute() else str((base / pp).resolve())

    paths = cfg.get("paths", {})
    for k, v in paths.items():
        if isinstance(v, str):
            paths[k] = _abs(v)

    cfg["paths"] = paths
    return cfg


def fmt_dur(sec: float) -> str:
    sec_i = int(sec)
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@contextlib.contextmanager
def stage_timer(name: str):
    t0 = time.perf_counter()
    print(f"[py] {name} START")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[py] {name} DONE in {dt:.3f}s ({fmt_dur(dt)})")
