#pragma once

#include <arm_acle.h>

template <typename KeyT>
struct CRCHash;

template <>
struct CRCHash<uint64_t> {
  size_t operator()(const uint64_t value) const noexcept {
    const uint32_t crc = __crc32cd(0, value);
    return static_cast<size_t>(crc) << 32 | crc;
  }
};
