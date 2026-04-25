#!/usr/bin/env python3

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


SMOG_ROOT = Path(__file__).resolve().parent
GRAPH_FOLDER = "../data/processed_graph/graph500-scale19-ef16/"
INPUT_PATTERNS = ["Q0", "Q1", "Q2", "Q3", "Q5"]
OUT_TXT = SMOG_ROOT / "preprocess_kernel_benchmark_results.txt"


_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _prepend_conda_bin_to_path() -> None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    bin_dir = os.path.join(prefix, "bin")
    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _run_case(
    cwd: Path, script: str, pattern: str, timeout_sec: Optional[int] = None
) -> Tuple[int, str]:
    cmd = [
        "python",
        script,
        "--input_graph_folder",
        GRAPH_FOLDER,
        "--input_pattern",
        pattern,
    ]
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=os.environ.copy(),
    )
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, text


def parse_hie_times(log: str) -> Tuple[str, str]:
    m_hier = list(
        re.finditer(
            rf"hier preprocessing time:\s*({_FLOAT})\s*ms,\s*kernel time:\s*({_FLOAT})\s*ms",
            log,
        )
    )
    if m_hier:
        last = m_hier[-1]
        return last.group(1), last.group(2)

    m_b = re.search(
        rf"build time \(GPU insert kernel only\) is :\s*({_FLOAT})\s*ms", log
    )
    m_s = re.search(rf"scan time \(GPU extract kernel\) is :\s*({_FLOAT})\s*ms", log)
    if m_b and m_s:
        pre = float(m_b.group(1)) + float(m_s.group(1))
        m_k = list(re.finditer(rf"Hierarchical time:\s*({_FLOAT})\s*ms", log))
        ker = m_k[-1].group(1) if m_k else "NA"
        return str(pre), ker
    return "NA", "NA"


def parse_normal_times(log: str) -> Tuple[str, str]:
    m = list(
        re.finditer(
            rf"normal preprocessing time:\s*({_FLOAT})\s*ms,\s*kernel time:\s*({_FLOAT})\s*ms",
            log,
        )
    )
    if m:
        last = m[-1]
        return last.group(1), last.group(2)
    return "NA", "NA"


def main() -> None:
    _prepend_conda_bin_to_path()

    rows: list[str] = []
    rows.append("# preprocess / kernel benchmark\n")
    rows.append(f"# graph: {GRAPH_FOLDER}\n")
    rows.append(f"# generated: {datetime.now().isoformat(timespec='seconds')}\n")
    rows.append(
        "# columns: variant, pattern, preprocess_time_ms, kernel_time_ms, exit_code\n"
    )
    rows.append("variant\tpattern\tpreprocess_time_ms\tkernel_time_ms\texit_code\n")

    cases = [
        ("hie", SMOG_ROOT / "SMOG_hie", "script_hie.py", parse_hie_times),
        ("normal", SMOG_ROOT / "SMOG_normal", "script.py", parse_normal_times),
    ]

    for variant, subdir, script, parser in cases:
        if not subdir.is_dir():
            rows.append(f"{variant}\t-\tNA\tNA\tMISSING_DIR\n")
            continue
        sc = subdir / script
        if not sc.is_file():
            rows.append(f"{variant}\t-\tNA\tNA\tMISSING_SCRIPT\n")
            continue

        for pat in INPUT_PATTERNS:
            print(f"[{variant}] {pat} ...", flush=True)
            code, log = _run_case(subdir, script, pat, timeout_sec=None)
            pre, ker = parser(log) if code == 0 else ("NA", "NA")
            rows.append(f"{variant}\t{pat}\t{pre}\t{ker}\t{code}\n")
            if code != 0:
                err_path = SMOG_ROOT / f"benchmark_fail_{variant}_{pat}.log"
                err_path.write_text(log, encoding="utf-8", errors="replace")
                print(f"  exit {code}, log -> {err_path}", flush=True)
            elif pre == "NA" and ker == "NA":
                dbg = SMOG_ROOT / f"benchmark_parse_debug_{variant}_{pat}.log"
                dbg.write_text(log[-200000:], encoding="utf-8", errors="replace")
                print(f"  parse NA, tail log -> {dbg}", flush=True)

    OUT_TXT.write_text("".join(rows), encoding="utf-8")
    print(f"Wrote {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
