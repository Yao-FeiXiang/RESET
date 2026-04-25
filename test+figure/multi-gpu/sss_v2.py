import os
import re
import csv
import io
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from time import perf_counter as clock

# =============================
# User-config
# =============================
APP = "../../multi-GPU/sss-multi-GPU/sss"
INPUT_FOLDER = "/data4/cliu26/arabic-2005/out"

NGPUS = [4, 8, 16, 32, 64]
ALPHA = 0.2
BUCKET = 5

KERNEL_REGEX = r"set_similarity_search_kernel"
NORMAL_FLAG = "--normal"

OUT_DIR = Path("./ncu_out").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

NCU_TMPROOT = Path("../../../tmp/ncu_tmp").resolve()
NCU_TMPROOT.mkdir(parents=True, exist_ok=True)

METRICS_MAP = {
    "gpu__time_duration.sum": "GPU_duration_ms",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1GlobalLoadSectors",
}
METRICS = list(METRICS_MAP.keys())
DUMP_STDOUT = True
STDOUT_TAIL_LINES = 140
STDERR_TAIL_LINES = 200
CSV_HEADER = [
    "NGPU",
    "normal",
    "alpha",
    "bucket",
    "kernel_instances",
    "GPU_duration_ms",
    "L1GlobalLoadSectors",
    "status",
    "error",
]


def _append_row(writer: csv.DictWriter, f, row: Dict[str, Any]) -> None:
    writer.writerow({h: row.get(h, "") for h in CSV_HEADER})
    f.flush()
    os.fsync(f.fileno())


def run(cmd: List[str], env_extra: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """执行外部命令并返回结果（不抛异常，交由调用方处理 returncode）。"""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=env
    )


def _tail_text(s: str, n: int) -> str:
    lines = s.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else s


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _time_to_ms(value: float, unit: str) -> float:
    u = (unit or "").strip().lower()
    if u in ("ns", "nsec"):
        return value / 1e6
    if u in ("us", "usec"):
        return value / 1e3
    if u in ("ms", "msec", ""):
        return value
    if u in ("s", "sec"):
        return value * 1e3
    return value


def parse_ncu_csv_summary(
    text: str, kernel_regex: str, metrics: List[str]
) -> Dict[str, Tuple[float, str]]:
    """
    解析 ncu --csv 的“汇总表”输出（可能前面夹杂程序输出/PROF行）：
    - 自动定位 CSV header 行
    - 按 Kernel Name 匹配
    - 对每个 Metric Name 取 Maximum 列
    返回：{metric_name: (max_value, unit)}，并附加 '__invocations__'。
    """
    lines_all = text.splitlines()

    # 定位 header 行：它通常以 "Process ID" 开头，并包含 "Kernel Name"、"Metric Name"、"Maximum"
    header_idx = None
    for i, ln in enumerate(lines_all):
        if (
            ('"Process ID"' in ln or "Process ID" in ln)
            and ("Kernel Name" in ln)
            and ("Metric Name" in ln)
            and ("Maximum" in ln)
        ):
            header_idx = i
            break

    if header_idx is None:
        return {}

    csv_lines = [ln for ln in lines_all[header_idx:] if ln.strip()]
    if len(csv_lines) < 2:
        return {}

    ker = re.compile(kernel_regex)
    metrics_set = set(metrics)

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    out: Dict[str, Tuple[float, str]] = {}
    invocations: Optional[int] = None

    for row in reader:
        kname = (row.get("Kernel Name") or "").strip()
        mname = (row.get("Metric Name") or "").strip()
        unit = (row.get("Metric Unit") or "").strip()
        mx = (row.get("Maximum") or "").strip()
        inv = (row.get("Invocations") or "").strip()

        if not kname or not mname:
            continue
        if not ker.search(kname):
            continue
        if mname not in metrics_set:
            continue

        if invocations is None and inv:
            try:
                invocations = int(float(inv.replace(",", "")))
            except Exception:
                invocations = None

        v = _to_float(mx)
        if v is None:
            continue
        out[mname] = (v, unit)

    if invocations is not None:
        out["__invocations__"] = (float(invocations), "")

    return out


def parse_instances(
    text: str, kernel_regex: str, metrics: List[str]
) -> List[Dict[str, Tuple[float, str]]]:
    """
    统一入口：优先解析 ncu --csv 汇总表；成功则返回单个 dict。
    """
    summary = parse_ncu_csv_summary(text, kernel_regex, metrics)
    return [summary] if summary else []


def run_ncu(total_device: int, enable_normal: bool, alpha: float, bucket: int) -> Dict[str, Any]:
    """运行一次 ncu profiling，解析汇总表并输出一行结果（使用 Maximum）。"""
    cmd_app = [
        APP,
        INPUT_FOLDER,
        f"--alpha={alpha}",
        f"--bucket={bucket}",
        f"--total_device={total_device}",
    ]
    if enable_normal:
        cmd_app.append(NORMAL_FLAG)

    cfg_tag = f"ngpu{total_device}_normal{int(enable_normal)}"
    tmpdir = NCU_TMPROOT / cfg_tag
    tmpdir.mkdir(parents=True, exist_ok=True)

    stdout_dump = OUT_DIR / f"ncu_stdout_{cfg_tag}.txt"

    cmd_ncu = [
        "ncu",
        "--kernel-name",
        f"regex:{KERNEL_REGEX}",
        "--csv",
        "--metrics",
        ",".join(METRICS),
        "--print-summary",
        "per-kernel",
        *cmd_app,
    ]

    cp = run(cmd_ncu, env_extra={"TMPDIR": str(tmpdir), "TMP": str(tmpdir), "TEMP": str(tmpdir)})

    if DUMP_STDOUT:
        try:
            stdout_dump.write_text(cp.stdout, errors="replace")
        except Exception:
            pass

    if cp.returncode != 0:
        raise RuntimeError(
            "NCU failed.\n"
            f"CMD: {' '.join(cmd_ncu)}\n"
            f"STDERR (tail):\n{_tail_text(cp.stderr, STDERR_TAIL_LINES)}\n"
            f"STDOUT (tail):\n{_tail_text(cp.stdout, STDOUT_TAIL_LINES)}\n"
            f"STDOUT dumped to: {stdout_dump}\n"
        )

    instances = parse_instances(cp.stdout, KERNEL_REGEX, METRICS)
    if not instances:
        raise RuntimeError(
            "No kernel instances parsed from NCU output.\n"
            f"Check kernel regex: {KERNEL_REGEX}\n"
            f"CMD: {' '.join(cmd_ncu)}\n"
            f"STDOUT (tail):\n{_tail_text(cp.stdout, STDOUT_TAIL_LINES)}\n"
            f"STDOUT dumped to: {stdout_dump}\n"
        )

    inst0 = instances[0]
    inv = inst0.get("__invocations__")
    kernel_instances = int(inv[0]) if inv else 0

    out: Dict[str, Any] = {
        "NGPU": total_device,
        "normal": int(enable_normal),
        "alpha": alpha,
        "bucket": bucket,
        "kernel_instances": kernel_instances,
    }

    for metric_name in METRICS:
        if metric_name not in inst0:
            continue
        val, unit = inst0[metric_name]
        col = METRICS_MAP[metric_name]
        if metric_name == "gpu__time_duration.sum":
            out[col] = _time_to_ms(val, unit)
        else:
            out[col] = val

    return out


def main():
    csv_path = OUT_DIR / "res.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

        for ngpu in NGPUS:
            for normal in (False, True):
                print(f"== Run: NGPU={ngpu}, normal={normal} ==")
                start = clock()
                try:
                    r = run_ncu(ngpu, normal, ALPHA, BUCKET)
                    r["status"] = "OK"
                    r["error"] = ""
                    _append_row(w, f, r)
                    elapsed = clock() - start
                    print(f"[ OK ] NGPU={ngpu}, normal={normal}: elapsed {elapsed:.2f} sec")

                except Exception as e:
                    fail_row = {
                        "NGPU": ngpu,
                        "normal": int(normal),
                        "alpha": ALPHA,
                        "bucket": BUCKET,
                        "kernel_instances": 0,
                        "GPU_duration_ms": "",
                        "L1GlobalLoadSectors": "",
                        "status": "FAIL",
                        "error": str(e)[:8000],
                    }
                    _append_row(w, f, fail_row)
                    print(f"[FAIL] NGPU={ngpu}, normal={normal}: {e}")

    print(f"\nDone. CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
