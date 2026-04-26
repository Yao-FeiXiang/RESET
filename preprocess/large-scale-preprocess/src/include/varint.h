#pragma once
#include <cstdint>
#include <cstdio>

#include "common.h"

// 读取无符号 LEB128 varint
inline uint64_t read_uvarint(FILE* f) {
  uint64_t x = 0;
  int shift = 0;
  while (true) {
    int c = std::fgetc(f);
    if (c == EOF) die("varint", "unexpected EOF");
    uint8_t b = static_cast<uint8_t>(c);
    x |= (uint64_t)(b & 0x7Fu) << shift;
    if ((b & 0x80u) == 0) break;
    shift += 7;
    if (shift > 63) die("varint", "uvarint too long");
  }
  return x;
}

// ZigZag 解码到有符号整数
inline int64_t zigzag_decode(uint64_t z) {
  return (int64_t)((z >> 1) ^ (~(z & 1) + 1));  // (z>>1) ^ -(z&1)
}

// 读取 ZigZag 编码的有符号 varint
inline int64_t read_svarint(FILE* f) {
  uint64_t z = read_uvarint(f);
  return zigzag_decode(z);
}

// 写入无符号 LEB128 varint
inline void write_uvarint(FILE* f, uint64_t x) {
  while (true) {
    uint8_t b = (uint8_t)(x & 0x7Fu);
    x >>= 7;
    if (x) {
      b |= 0x80u;
      std::fputc((int)b, f);
    } else {
      std::fputc((int)b, f);
      break;
    }
  }
}

// ZigZag 编码
inline uint64_t zigzag_encode(int64_t x) {
  return (uint64_t)((x << 1) ^ (x >> 63));
}

// 写入 ZigZag 编码的有符号 varint
inline void write_svarint(FILE* f, int64_t x) {
  write_uvarint(f, zigzag_encode(x));
}
