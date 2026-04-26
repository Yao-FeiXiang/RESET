import os
import struct
from typing import Optional


def check_parts_exist(parts: list[str], kind: str) -> None:
    for p in parts:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {kind} part: {p}")


def validate_vector_file(path: str, elem_size: int, fmt_char: str, *, max_print: int = 0) -> bool:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    sz = os.path.getsize(path)
    if sz < 8:
        return False

    with open(path, "rb") as f:
        hdr = f.read(8)
        if len(hdr) != 8:
            return False
        (n,) = struct.unpack("<Q", hdr)

        expected = 8 + elem_size * int(n)
        if expected != sz:
            return False

        if max_print > 0 and n > 0:
            k = min(int(n), max_print)
            data = f.read(elem_size * k)
            if len(data) != elem_size * k:
                return False
            vals = struct.unpack("<" + fmt_char * k, data)
            print(f"[fmt] {path}: size={n}, head({k})={vals}")

    return True


def validate_vector_int32(path: str, *, max_print: int = 0) -> bool:
    return validate_vector_file(path, elem_size=4, fmt_char="i", max_print=max_print)


def validate_vector_int64(path: str, *, max_print: int = 0) -> bool:
    return validate_vector_file(path, elem_size=8, fmt_char="q", max_print=max_print)


# -------------------------
# Writing helpers
# -------------------------
def count_u32_elements(paths: list[str]) -> int:
    total = 0
    for p in paths:
        s = os.path.getsize(p)
        if s % 4 != 0:
            raise ValueError(f"bad u32 file (size%4!=0): {p}")
        total += s // 4
    return total


def write_int32_vec(out_path: str, inputs: list[str]) -> None:
    """
    Write vector<int32>:
      [u64 size] + [int32...]
    Payload is copied from u32 part files as-is (byte-for-byte).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = count_u32_elements(inputs)

    with open(out_path, "wb") as w:
        w.write(struct.pack("<Q", int(total)))
        for p in inputs:
            with open(p, "rb") as r:
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    w.write(b)


def merge_offsets_int64(
    out_offsets_bin: str,
    off_parts: list[str],
    expected_n: int,
    progress_cb: Optional[callable] = None,
) -> int:
    os.makedirs(os.path.dirname(out_offsets_bin), exist_ok=True)

    prefix = 0
    written_n = 0
    with open(out_offsets_bin, "wb") as out:
        out.write(struct.pack("<Q", int(expected_n) + 1))
        out.write(struct.pack("<q", 0))  # offsets[0]

        total = len(off_parts)
        for i, offp in enumerate(off_parts):
            if progress_cb:
                progress_cb(i, total)

            data = open(offp, "rb").read()
            if len(data) % 4 != 0:
                raise ValueError(f"bad u32 offsets file: {offp}")
            cnt = len(data) // 4
            if cnt < 1:
                raise ValueError(f"empty offsets file: {offp}")

            local = struct.unpack("<" + "I" * cnt, data)
            if local[0] != 0:
                raise ValueError(f"offsets part does not start with 0: {offp}")

            for x in local[1:]:
                v = prefix + int(x)
                out.write(struct.pack("<q", int(v)))

            prefix += int(local[-1])
            written_n += cnt - 1

    if written_n != int(expected_n):
        raise ValueError(
            f"global offsets length mismatch: wrote n={written_n}, expected n={int(expected_n)}"
        )

    return prefix


def merge_partition_outputs(
    cols_bin: str,
    offsets_bin: str,
    col_parts: list[str],
    off_parts: list[str],
    n: int,
    vertexs_bin: Optional[str] = None,
    ver_parts: Optional[list[str]] = None,
    log_every: Optional[int] = None,
) -> int:
    # cols -> vector<int32>
    write_int32_vec(cols_bin, col_parts)

    total = len(off_parts)
    if log_every is None:
        log_every = max(1, total // 10)

    def _progress(i: int, total: int) -> None:
        if i % log_every == 0 or i == total - 1:
            print(f"[py][Stage3] offsets progress: part {i + 1}/{total}")

    # offsets -> vector<int64>
    total_edges = merge_offsets_int64(
        offsets_bin, off_parts, expected_n=int(n), progress_cb=_progress
    )

    # vertexs -> vector<int32>
    if vertexs_bin is not None:
        if not ver_parts:
            raise ValueError("vertexs_bin provided but ver_parts is empty")
        write_int32_vec(vertexs_bin, ver_parts)

    if not validate_vector_int32(cols_bin):
        raise RuntimeError(f"bad output format for cols (vec<i32>): {cols_bin}")
    if not validate_vector_int64(offsets_bin):
        raise RuntimeError(f"bad output format for offsets (vec<i64>): {offsets_bin}")
    if vertexs_bin is not None and not validate_vector_int32(vertexs_bin):
        raise RuntimeError(f"bad output format for vertexs (vec<i32>): {vertexs_bin}")

    return total_edges
