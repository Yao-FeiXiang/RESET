#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace fs = std::filesystem;

// 简单日志打印
inline void log_line(const std::string& tag, const std::string& msg) {
  std::cerr << "[" << tag << "] " << msg << "\n";
}

// 错误后退出
[[noreturn]] inline void die(const std::string& tag, const std::string& msg) {
  std::cerr << "[" << tag << "][ERR] " << msg << "\n";
  std::exit(1);
}

// 读取 8 字节小端 u64 文件
inline uint64_t read_u64_file(const fs::path& p) {
  FILE* f = std::fopen(p.string().c_str(), "rb");
  if (!f) die("io", "open failed: " + p.string());
  uint64_t x = 0;
  if (std::fread(&x, 1, 8, f) != 8) die("io", "read failed: " + p.string());
  std::fclose(f);
  return x;
}

// 写入 8 字节小端 u64 文件
inline void write_u64_file(const fs::path& p, uint64_t x) {
  FILE* f = std::fopen(p.string().c_str(), "wb");
  if (!f) die("io", "open failed: " + p.string());
  if (std::fwrite(&x, 1, 8, f) != 8) die("io", "write failed: " + p.string());
  std::fclose(f);
}

// 确保目录存在
inline void ensure_dir(const fs::path& p) {
  std::error_code ec;
  fs::create_directories(p, ec);
  if (ec) die("io", "mkdir failed: " + p.string() + " : " + ec.message());
}

// 读取整个文本文件
inline std::string read_text_file(const fs::path& p) {
  FILE* f = std::fopen(p.string().c_str(), "rb");
  if (!f) die("io", "open failed: " + p.string());
  std::string s;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  if (n < 0) die("io", "ftell failed: " + p.string());
  std::fseek(f, 0, SEEK_SET);
  s.resize((size_t)n);
  if (n > 0 && std::fread(s.data(), 1, (size_t)n, f) != (size_t)n) {
    die("io", "read failed: " + p.string());
  }
  std::fclose(f);
  return s;
}

// =======================
// 极简 JSON 取值工具（适配本项目 config.json）
// 约束：key 必须双引号；值是 string/number/bool；不支持数组/复杂嵌套解析。
// path 用 "a.b.c" 形式。
// =======================
inline size_t find_quoted_key(const std::string& j, const std::string& key,
                              size_t from = 0) {
  std::string pat = "\"" + key + "\"";
  return j.find(pat, from);
}

inline size_t find_colon_after(const std::string& j, size_t pos) {
  size_t c = j.find(':', pos);
  if (c == std::string::npos) return c;
  return c;
}

inline size_t skip_ws(const std::string& j, size_t i) {
  while (i < j.size()) {
    char ch = j[i];
    if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t')
      i++;
    else
      break;
  }
  return i;
}

// 从一个对象片段中找 key 的值起点（返回冒号后第一个非空白位置）
inline std::optional<size_t> locate_value_pos_in_object(
    const std::string& j, size_t obj_lbrace, const std::string& key) {
  size_t k = find_quoted_key(j, key, obj_lbrace);
  if (k == std::string::npos) return std::nullopt;
  size_t c = find_colon_after(j, k);
  if (c == std::string::npos) return std::nullopt;
  return skip_ws(j, c + 1);
}

// 返回某个对象 key（比如 "paths"）对应对象的 '{' 位置
inline std::optional<size_t> locate_object_lbrace(const std::string& j,
                                                  size_t from,
                                                  const std::string& obj_key) {
  size_t k = find_quoted_key(j, obj_key, from);
  if (k == std::string::npos) return std::nullopt;
  size_t c = find_colon_after(j, k);
  if (c == std::string::npos) return std::nullopt;
  size_t v = skip_ws(j, c + 1);
  if (v >= j.size() || j[v] != '{') return std::nullopt;
  return v;
}

// 按 "a.b.c" 逐级定位对象
inline std::optional<size_t> locate_parent_object(
    const std::string& j, const std::string& dotted_path) {
  // 返回最后一级 key 所在对象的 '{'
  size_t cur_obj = 0;

  // 根对象必须从第一个 '{' 开始
  cur_obj = j.find('{');
  if (cur_obj == std::string::npos) return std::nullopt;

  size_t start = 0;
  while (true) {
    size_t dot = dotted_path.find('.', start);
    if (dot == std::string::npos) break;
    std::string seg = dotted_path.substr(start, dot - start);
    auto ob = locate_object_lbrace(j, cur_obj, seg);
    if (!ob) return std::nullopt;
    cur_obj = *ob;
    start = dot + 1;
  }
  return cur_obj;
}

inline std::string cfg_get_string(const std::string& j,
                                  const std::string& path) {
  auto parent = locate_parent_object(j, path);
  if (!parent) die("cfg", "bad config path: " + path);
  std::string key = path.substr(path.find_last_of('.') == std::string::npos
                                    ? 0
                                    : path.find_last_of('.') + 1);
  auto vpos = locate_value_pos_in_object(j, *parent, key);
  if (!vpos) die("cfg", "missing key: " + path);
  size_t i = *vpos;
  if (i >= j.size() || j[i] != '"') die("cfg", "expected string for: " + path);
  size_t end = j.find('"', i + 1);
  if (end == std::string::npos) die("cfg", "unterminated string for: " + path);
  return j.substr(i + 1, end - (i + 1));
}

inline uint64_t cfg_get_u64(const std::string& j, const std::string& path) {
  auto parent = locate_parent_object(j, path);
  if (!parent) die("cfg", "bad config path: " + path);
  std::string key = path.substr(path.find_last_of('.') == std::string::npos
                                    ? 0
                                    : path.find_last_of('.') + 1);
  auto vpos = locate_value_pos_in_object(j, *parent, key);
  if (!vpos) die("cfg", "missing key: " + path);
  size_t i = *vpos;
  // 读到数字结束（逗号/右括号/空白）
  size_t e = i;
  while (e < j.size()) {
    char ch = j[e];
    if ((ch >= '0' && ch <= '9'))
      e++;
    else
      break;
  }
  if (e == i) die("cfg", "expected number for: " + path);
  return std::stoull(j.substr(i, e - i));
}

inline bool cfg_get_bool(const std::string& j, const std::string& path) {
  auto parent = locate_parent_object(j, path);
  if (!parent) die("cfg", "bad config path: " + path);
  std::string key = path.substr(path.find_last_of('.') == std::string::npos
                                    ? 0
                                    : path.find_last_of('.') + 1);
  auto vpos = locate_value_pos_in_object(j, *parent, key);
  if (!vpos) die("cfg", "missing key: " + path);
  size_t i = *vpos;
  if (j.compare(i, 4, "true") == 0) return true;
  if (j.compare(i, 5, "false") == 0) return false;
  die("cfg", "expected bool for: " + path);
  return false;
}

inline std::string load_config_text() {
  const char* env = std::getenv("CONFIG");
  fs::path p = env ? fs::path(env) : fs::path("config.json");
  return read_text_file(p);
}
