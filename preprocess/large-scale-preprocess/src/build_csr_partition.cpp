#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <vector>

#include "common.h"

#pragma pack(push, 1)
struct EdgePair32 {
  uint32_t u;
  uint32_t v;
};
#pragma pack(pop)

static inline uint32_t part_begin(uint32_t part_id, uint32_t parts,
                                  uint32_t n) {
  return (uint32_t)(((uint64_t)part_id * n) / parts);
}
static inline uint32_t part_end(uint32_t part_id, uint32_t parts, uint32_t n) {
  return (uint32_t)(((uint64_t)(part_id + 1) * n) / parts);
}

static inline uint32_t get_part_id() {
  const char* s = std::getenv("PART_ID");
  if (!s) die("build", "missing env PART_ID");
  long v = std::strtol(s, nullptr, 10);
  if (v < 0) die("build", "bad PART_ID");
  return (uint32_t)v;
}

static inline void write_u32_vec(const fs::path& p,
                                 const std::vector<uint32_t>& a) {
  FILE* f = std::fopen(p.string().c_str(), "wb");
  if (!f) die("build", "open failed: " + p.string());
  if (!a.empty()) {
    if (std::fwrite(a.data(), sizeof(uint32_t), a.size(), f) != a.size())
      die("build", "write failed: " + p.string());
  }
  std::fclose(f);
}

// 追加写 u32（用于流式写 cols/vertexs）
static inline void append_u32(FILE* f, uint32_t x) {
  if (std::fwrite(&x, sizeof(uint32_t), 1, f) != 1)
    die("build", "write failed");
}

// run 读取器
struct RunReader {
  FILE* f = nullptr;
  EdgePair32 cur{};
  bool has = false;

  explicit RunReader(const fs::path& p) {
    f = std::fopen(p.string().c_str(), "rb");
    if (!f) die("build", "open run failed: " + p.string());
    advance();
  }
  ~RunReader() {
    if (f) std::fclose(f);
    f = nullptr;
  }
  void advance() {
    size_t r = std::fread(&cur, sizeof(cur), 1, f);
    has = (r == 1);
  }
};

struct HeapItem {
  EdgePair32 e;
  int rid;
};

struct HeapCmp {
  bool operator()(const HeapItem& a, const HeapItem& b) const {
    if (a.e.u != b.e.u) return a.e.u > b.e.u;
    return a.e.v > b.e.v;
  }
};

int main() {
  const std::string cfg = load_config_text();

  fs::path work_dir = cfg_get_string(cfg, "paths.work_dir");
  fs::path part_dir = work_dir / "parts";
  fs::path tmp_dir = work_dir / "tmp";
  ensure_dir(tmp_dir);

  uint32_t parts = (uint32_t)cfg_get_u64(cfg, "partition.num_parts");
  bool verbose = cfg_get_bool(cfg, "runtime.verbose");

  uint64_t n64 = read_u64_file(part_dir / "num_nodes.bin");
  if (n64 > 0xFFFFFFFFULL) die("build", "n too large for u32");
  uint32_t n = (uint32_t)n64;

  std::string mode = cfg_get_string(cfg, "pipeline.mode");  // "sss" or "tc"
  uint64_t chunk_edges64 = cfg_get_u64(cfg, "pipeline.chunk_edges");
  if (chunk_edges64 == 0) die("build", "pipeline.chunk_edges must be >0");
  if (chunk_edges64 > (1ULL << 31)) die("build", "chunk_edges too large");
  size_t chunk_edges = (size_t)chunk_edges64;

  uint32_t pid = get_part_id();
  if (pid >= parts) die("build", "PART_ID out of range");

  uint32_t u0 = part_begin(pid, parts, n);
  uint32_t u1 = part_end(pid, parts, n);
  uint32_t local_n = u1 - u0;

  fs::path in_path = part_dir / ("part_" + std::to_string(pid) + ".bin");
  FILE* in = std::fopen(in_path.string().c_str(), "rb");
  if (!in) die("build", "open failed: " + in_path.string());

  std::vector<fs::path> runs;
  std::vector<EdgePair32> buf;
  buf.reserve(chunk_edges);

  uint64_t edges_in = 0;
  uint64_t edges_kept = 0;

  while (true) {
    buf.clear();
    buf.shrink_to_fit();
    buf.reserve(chunk_edges);

    for (size_t i = 0; i < chunk_edges; i++) {
      EdgePair32 e;
      size_t r = std::fread(&e, sizeof(e), 1, in);
      if (r == 0) break;
      edges_in++;
      if (e.u == e.v) continue;
      if (e.u < u0 || e.u >= u1) continue;

      buf.push_back(e);
      edges_kept++;
    }

    if (buf.empty()) {
      if (std::feof(in)) break;
      if (std::ferror(in)) die("build", "read error");
      if (std::ftell(in) == 0) die("build", "unexpected empty input");
      if (std::feof(in)) break;
    }

    if (!buf.empty()) {
      std::sort(buf.begin(), buf.end(),
                [](const EdgePair32& a, const EdgePair32& b) {
                  if (a.u != b.u) return a.u < b.u;
                  return a.v < b.v;
                });

      fs::path run_p = tmp_dir / ("run_" + std::to_string(pid) + "_" +
                                  std::to_string(runs.size()) + ".bin");
      FILE* rf = std::fopen(run_p.string().c_str(), "wb");
      if (!rf) die("build", "open run for write failed: " + run_p.string());
      if (std::fwrite(buf.data(), sizeof(EdgePair32), buf.size(), rf) !=
          buf.size())
        die("build", "write run failed: " + run_p.string());
      std::fclose(rf);

      runs.push_back(run_p);
    }

    if (std::feof(in)) break;
  }

  std::fclose(in);

  if (runs.empty()) {
    std::vector<uint32_t> csr_offsets(local_n + 1, 0);
    fs::path off_p =
        tmp_dir / ("csr_offsets_part_" + std::to_string(pid) + ".bin");
    fs::path col_p =
        tmp_dir / ("csr_cols_part_" + std::to_string(pid) + ".bin");
    write_u32_vec(off_p, csr_offsets);
    write_u32_vec(col_p, std::vector<uint32_t>{});

    if (mode == "sss") {
      fs::path ver_p =
          tmp_dir / ("vertexs_part_" + std::to_string(pid) + ".bin");
      write_u32_vec(ver_p, std::vector<uint32_t>{});
    }

    write_u64_file(tmp_dir / ("num_nodes_part_" + std::to_string(pid) + ".bin"),
                   local_n);
    write_u64_file(tmp_dir / ("num_edges_part_" + std::to_string(pid) + ".bin"),
                   0);

    if (verbose) {
      log_line("build",
               "pid=" + std::to_string(pid) +
                   " empty partition, local_n=" + std::to_string(local_n));
    }
    return 0;
  }

  // ===== Phase2: k-way merge runs -> stream CSR =====
  std::vector<RunReader*> readers;
  readers.reserve(runs.size());
  for (auto& p : runs) readers.push_back(new RunReader(p));

  std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> pq;
  for (int i = 0; i < (int)readers.size(); i++) {
    if (readers[i]->has) pq.push(HeapItem{readers[i]->cur, i});
  }

  fs::path off_p =
      tmp_dir / ("csr_offsets_part_" + std::to_string(pid) + ".bin");
  fs::path col_p = tmp_dir / ("csr_cols_part_" + std::to_string(pid) + ".bin");
  fs::path ver_p = tmp_dir / ("vertexs_part_" + std::to_string(pid) + ".bin");

  std::vector<uint32_t> csr_offsets(local_n + 1, 0);

  FILE* col_f = std::fopen(col_p.string().c_str(), "wb");
  if (!col_f) die("build", "open cols out failed: " + col_p.string());
  std::setvbuf(col_f, nullptr, _IOFBF, 1 << 20);

  FILE* ver_f = nullptr;
  bool use_vertexs = (mode == "sss");
  if (use_vertexs) {
    ver_f = std::fopen(ver_p.string().c_str(), "wb");
    if (!ver_f) die("build", "open vertexs out failed: " + ver_p.string());
    std::setvbuf(ver_f, nullptr, _IOFBF, 1 << 20);
  }

  uint64_t edges_out = 0;

  uint32_t cur_u = u0;
  uint32_t cur_local = 0;

  bool has_last_v = false;
  uint32_t last_v = 0;

  auto finalize_u = [&](uint32_t u_global) {
    (void)u_global;
    csr_offsets[cur_local + 1] = (uint32_t)edges_out;
    cur_local++;
    cur_u++;
    has_last_v = false;
  };

  while (!pq.empty()) {
    HeapItem it = pq.top();
    pq.pop();

    EdgePair32 e = it.e;
    RunReader* rr = readers[it.rid];
    rr->advance();
    if (rr->has) pq.push(HeapItem{rr->cur, it.rid});

    if (e.u < u0 || e.u >= u1) continue;
    while (cur_u < e.u && cur_local < local_n) {
      finalize_u(cur_u);
    }

    if (cur_u != e.u) {
      if (cur_u > e.u) continue;
    }

    uint32_t u_global = e.u;
    uint32_t v = e.v;
    if (has_last_v && v == last_v) continue;
    if (mode == "tc") {
      if (v <= u_global) {
        last_v = v;
        has_last_v = true;
        continue;
      }
    }
    append_u32(col_f, v);
    if (use_vertexs) append_u32(ver_f, u_global);

    edges_out++;
    last_v = v;
    has_last_v = true;
  }

  while (cur_local < local_n) {
    finalize_u(cur_u);
  }

  std::fclose(col_f);
  if (ver_f) std::fclose(ver_f);

  write_u32_vec(off_p, csr_offsets);
  write_u64_file(tmp_dir / ("num_nodes_part_" + std::to_string(pid) + ".bin"),
                 local_n);
  write_u64_file(tmp_dir / ("num_edges_part_" + std::to_string(pid) + ".bin"),
                 edges_out);
  for (auto* p : readers) delete p;
  for (auto& p : runs) {
    std::error_code ec;
    fs::remove(p, ec);
  }

  if (verbose) {
    log_line("build",
             "pid=" + std::to_string(pid) + " u_range=[" + std::to_string(u0) +
                 "," + std::to_string(u1) + ")" +
                 " local_n=" + std::to_string(local_n) +
                 " edges_in=" + std::to_string(edges_in) +
                 " edges_kept(u-in-range)=" + std::to_string(edges_kept) +
                 " edges_out(unique" + std::string(mode == "tc" ? "+tc" : "") +
                 ")=" + std::to_string(edges_out) +
                 " runs=" + std::to_string(runs.size()) + " mode=" + mode);
  }

  return 0;
}
