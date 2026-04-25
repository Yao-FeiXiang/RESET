#pragma once
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "common.h"
#include "varint.h"

struct BCSRReader {
  fs::path in_dir;
  uint64_t n = 0;
  uint64_t m = 0;
  uint64_t B = 0;
  uint64_t num_blocks = 0;
  std::vector<uint64_t> block_offsets;
  FILE* cols = nullptr;

  uint64_t cur_u = 0;
  uint64_t cur_block = 0;
  uint64_t cur_block_end_u = 0;

  static inline int64_t file_tell64(FILE* f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return (int64_t)std::ftell(f);
#endif
  }

  static inline int file_seek64(FILE* f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (int64_t)off, SEEK_SET);
#else
    return std::fseek(f, (long)off, SEEK_SET);
#endif
  }

  BCSRReader(const fs::path& dir, uint64_t block_nodes)
      : in_dir(dir), B(block_nodes) {
    n = read_u64_file(in_dir / "num_nodes.bin");
    m = read_u64_file(in_dir / "num_edges.bin");
    num_blocks = (n + B - 1) / B;

    // ---- read offsets ----
    fs::path off_p = in_dir / "raw_csr_offsets.bin";
    FILE* off_f = std::fopen(off_p.string().c_str(), "rb");
    if (!off_f) die("bcsr", "open offsets failed: " + off_p.string());

    std::error_code ec;
    uint64_t off_bytes = (uint64_t)fs::file_size(off_p, ec);
    if (ec)
      die("bcsr",
          "stat offsets failed: " + off_p.string() + " : " + ec.message());

    uint64_t expected_bytes = (num_blocks + 1) * 8ULL;
    if (off_bytes != expected_bytes) {
      die("bcsr",
          "offsets size mismatch: file_bytes=" + std::to_string(off_bytes) +
              " expected_bytes=" + std::to_string(expected_bytes) +
              " (n=" + std::to_string(n) + " B=" + std::to_string(B) +
              " num_blocks=" + std::to_string(num_blocks) + ")");
    }

    block_offsets.resize(num_blocks + 1);
    size_t got =
        std::fread(block_offsets.data(), 8, (size_t)(num_blocks + 1), off_f);
    if (got != (size_t)(num_blocks + 1)) {
      die("bcsr", "read offsets failed: got=" + std::to_string(got) +
                      " expected=" + std::to_string(num_blocks + 1));
    }
    std::fclose(off_f);

    // ---- open cols ----
    fs::path col_p = in_dir / "raw_csr_cols.bin";
    cols = std::fopen(col_p.string().c_str(), "rb");
    if (!cols) die("bcsr", "open cols failed: " + col_p.string());

    cur_u = 0;
    cur_block = 0;
    cur_block_end_u = std::min<uint64_t>(B, n);

    if (file_seek64(cols, block_offsets[0]) != 0) {
      die("bcsr", "seek cols failed");
    }
  }

  ~BCSRReader() {
    if (cols) std::fclose(cols);
    cols = nullptr;
  }

  template <class EdgeFn>
  bool next_node(EdgeFn&& fn_edge) {
    if (cur_u >= n) return false;

    if (cur_u == cur_block_end_u) {
      cur_block++;
      if (cur_block >= num_blocks) return false;

      uint64_t start_u = cur_block * B;
      cur_block_end_u = std::min<uint64_t>((cur_block + 1) * B, n);

      if (file_seek64(cols, block_offsets[cur_block]) != 0) {
        die("bcsr", "seek cols failed at block " + std::to_string(cur_block));
      }
      if (cur_u != start_u) die("bcsr", "node index mismatch");
    }

    uint64_t block_end_off = block_offsets[cur_block + 1];
    int64_t pos = file_tell64(cols);
    if (pos < 0) die("bcsr", "ftell failed");

    if ((uint64_t)pos == block_end_off) {
      return false;
    }
    if ((uint64_t)pos > block_end_off) {
      die("bcsr",
          "cols position past block end: pos=" + std::to_string((uint64_t)pos) +
              " block_end=" + std::to_string(block_end_off) +
              " block=" + std::to_string(cur_block));
    }

    uint64_t u = cur_u++;
    uint64_t deg = read_uvarint(cols);

    uint64_t prev = 0;
    for (uint64_t i = 0; i < deg; i++) {
      int64_t delta = read_svarint(cols);
      int64_t vv = (int64_t)prev + delta;
      if (vv < 0) die("bcsr", "negative neighbor id after decode");
      uint64_t v = (uint64_t)vv;
      prev = v;
      fn_edge(u, v);
    }
    return true;
  }
};
