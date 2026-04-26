#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bcsr_reader.h"
#include "common.h"

#pragma pack(push, 1)
struct EdgePair32 {
  uint32_t u;
  uint32_t v;
};
#pragma pack(pop)

static inline uint32_t part_of(uint32_t u, uint32_t parts, uint32_t n) {
  uint64_t x = (uint64_t)u * parts;
  return (uint32_t)(x / n);
}

static inline uint64_t gcd_u64(uint64_t a, uint64_t b) {
  while (b) {
    uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

int main() {
  const std::string cfg = load_config_text();

  fs::path in_dir = cfg_get_string(cfg, "paths.bcsr_dir");
  fs::path work_dir = cfg_get_string(cfg, "paths.work_dir");
  fs::path out_dir = work_dir / "parts";

  uint64_t block_nodes = cfg_get_u64(cfg, "bcsr.block_nodes");
  uint32_t parts = (uint32_t)cfg_get_u64(cfg, "partition.num_parts");
  uint64_t step = cfg_get_u64(cfg, "partition.step");
  bool verbose = cfg_get_bool(cfg, "runtime.verbose");

  ensure_dir(out_dir);

  BCSRReader rd(in_dir, block_nodes);
  uint32_t n = (uint32_t)rd.n;

  // 保证 perm 是双射
  uint64_t step_orig = step;
  while (gcd_u64(step, (uint64_t)n) != 1) step++;
  if (verbose && step != step_orig) {
    log_line("extract",
             "adjust step to be coprime with n: " + std::to_string(step_orig) +
                 " -> " + std::to_string(step));
  }

  if (verbose) {
    log_line("extract", "n=" + std::to_string(rd.n) +
                            " m=" + std::to_string(rd.m) +
                            " parts=" + std::to_string(parts) +
                            " step=" + std::to_string(step));
  }

  std::vector<FILE*> fps(parts, nullptr);
  for (uint32_t p = 0; p < parts; p++) {
    fs::path path = out_dir / ("part_" + std::to_string(p) + ".bin");
    fps[p] = std::fopen(path.string().c_str(), "wb");
    if (!fps[p]) die("extract", "open failed: " + path.string());
    std::setvbuf(fps[p], nullptr, _IOFBF, 1 << 20);
  }

  uint64_t edges_out = 0;
  uint64_t nodes = 0;

  auto perm = [&](uint32_t x) -> uint32_t {
    return (uint32_t)(((uint64_t)x * step) % (uint64_t)n);
  };

  while (rd.next_node([&](uint64_t u0, uint64_t v0) {
    uint32_t u = perm((uint32_t)u0);
    uint32_t v = perm((uint32_t)v0);

    // 过滤自环
    if (u == v) return;

    // undirected: (u,v) and (v,u)
    uint32_t p1 = part_of(u, parts, n);
    uint32_t p2 = part_of(v, parts, n);

    EdgePair32 e1{u, v};
    EdgePair32 e2{v, u};

    if (std::fwrite(&e1, sizeof(e1), 1, fps[p1]) != 1)
      die("extract", "write failed");
    if (std::fwrite(&e2, sizeof(e2), 1, fps[p2]) != 1)
      die("extract", "write failed");
    edges_out += 2;
  })) {
    nodes++;
    if (verbose && (nodes % 1000000ULL) == 0) {
      log_line("extract", "processed nodes=" + std::to_string(nodes) + "/" +
                              std::to_string(rd.n) +
                              " edges_out=" + std::to_string(edges_out));
    }
  }

  for (auto* f : fps) std::fclose(f);

  write_u64_file(out_dir / "num_nodes.bin", rd.n);
  write_u64_file(out_dir / "num_edges_undirected.bin", edges_out);

  if (verbose) {
    log_line("extract", "done nodes=" + std::to_string(nodes) + "/" +
                            std::to_string(rd.n) +
                            " edges_out=" + std::to_string(edges_out));
  }
  return 0;
}
