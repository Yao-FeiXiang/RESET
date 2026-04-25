import os
import struct
import numpy as np
from typing import List, Optional, Sequence, Tuple
import pandas as pd


def merge_res(solver_res, runner_res):
    solver_map = {}
    for row in solver_res or []:
        if "alpha" not in row or "b" not in row:
            raise KeyError(f"Missing key in solver_res row: {row}")
        solver_map[(row["alpha"], row["b"])] = row

    merged = []
    for row in runner_res or []:
        if "alpha" not in row or "b" not in row or "table_type" not in row:
            raise KeyError(f"Missing key in runner_res row: {row}")

        key = (row["alpha"], row["b"])
        merged_row = {}

        if key in solver_map:
            merged_row.update(solver_map[key])
        merged_row.update(row)
        merged.append(merged_row)
    return merged


def auto_boundaries(
    pivot: pd.DataFrame,
    n_bins: int = 10,
    clip_q: Tuple[float, float] = (0.02, 0.98),
    mode: str = "quantile",  # "quantile" | "log-quantile" | "linear"
    min_step: float = 1e-12,
    keep_clip: bool = True,
) -> List[float]:
    x = pivot.to_numpy().astype(float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [0.0, 1.0]

    lo_q, hi_q = clip_q
    lo_q = float(np.clip(lo_q, 0.0, 1.0))
    hi_q = float(np.clip(hi_q, 0.0, 1.0))
    if hi_q <= lo_q:
        lo_q, hi_q = 0.02, 0.98

    lo = float(np.quantile(x, lo_q))
    hi = float(np.quantile(x, hi_q))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo == hi:
        lo, hi = float(np.min(x)), float(np.max(x))
        if lo == hi:
            return [lo, lo + 1.0]

    if mode == "linear":
        edges = np.linspace(lo, hi, n_bins + 1)

    elif mode == "log-quantile":
        xp = x[(x > 0) & np.isfinite(x)]
        if xp.size < 2:
            edges = np.linspace(lo, hi, n_bins + 1)
        else:
            llo = float(np.quantile(xp, lo_q))
            lhi = float(np.quantile(xp, hi_q))
            llo = max(llo, min_step)
            lhi = max(lhi, llo + min_step)

            q = np.linspace(lo_q, hi_q, n_bins + 1)
            edges = np.exp(np.quantile(np.log(xp), q))
            edges[0], edges[-1] = llo, lhi

    else:  # "quantile"
        edges = np.quantile(x, np.linspace(lo_q, hi_q, n_bins + 1))

    edges = np.asarray(edges, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))

    if edges.size < 2:
        lo2, hi2 = float(np.min(x)), float(np.max(x))
        return [lo2, hi2] if lo2 != hi2 else [lo2, lo2 + 1.0]

    if not keep_clip:
        data_min, data_max = float(np.min(x)), float(np.max(x))
        if edges[0] > data_min:
            edges = np.concatenate([[data_min], edges])
        if edges[-1] < data_max:
            edges = np.concatenate([edges, [data_max]])

    out = [float(edges[0])]
    for v in edges[1:]:
        v = float(v)
        if v <= out[-1]:
            v = out[-1] + min_step
        out.append(v)

    return out


def compute_arr_size(dataset_path: str, mode: str, version: str = "v2"):
    """
    v1:平均度
    v2:加权平均度
    """

    if mode in ("sss", "tc"):
        return _compute_graph(dataset_path, version=version)

    if mode == "ir":
        return _compute_ir(dataset_path, version=version)

    raise ValueError(f"Unknown mode: {mode}")


def read_bin(file_path: str):
    print(f"Reading file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        if file_size == struct.calcsize("i"):
            return struct.unpack("i", f.read(4))[0]
        size_bytes = f.read(struct.calcsize("Q"))
        size = struct.unpack("Q", size_bytes)[0]

        data = np.frombuffer(f.read(size * struct.calcsize("i")), dtype=np.int32, count=size)
        return data


def _compute_graph(dataset_path: str, version):
    offset_path = os.path.join(dataset_path, "csr_offsets.bin")
    offsets = read_bin(offset_path)
    degrees = offsets[1:] - offsets[:-1]

    if version == "v1":
        arr = degrees.mean()

    elif version == "v2":
        weights = degrees
        arr = (degrees * weights).sum() / max(1, weights.sum())

    else:
        raise ValueError(f"Unknown version: {version}")

    return max(1, int(arr))


def _compute_ir(dataset_path: str, version):
    offset_path = os.path.join(dataset_path, "query_offset.bin")
    offsets = read_bin(offset_path)

    lengths = offsets[1:] - offsets[:-1]

    if version == "v1":
        arr = lengths.mean()

    elif version == "v2":
        weights = lengths
        arr = (lengths * weights).sum() / max(1, weights.sum())

    else:
        raise ValueError(f"Unknown version: {version}")

    return max(1, int(arr))


def test():
    dataset_path = "../../graph_datasets/twitter"
    arr_size = compute_arr_size(dataset_path, mode="sss", version="v2")
    print(f"Computed arr size: {arr_size}")


if __name__ == "__main__":
    test()
