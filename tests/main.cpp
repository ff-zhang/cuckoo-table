#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "cuckoo_set.hpp"
#include "hash.hpp"
#include "huge_page_allocator.hpp"

constexpr size_t CAPACITY = 128 * 1024 * 1024;
constexpr size_t LOAD_PERCENTAGE = 80;
constexpr size_t HIT_PERCENTAGE = 80;
constexpr size_t NUM_REQUESTS = 100000000;
constexpr size_t NUM_KEYS = CAPACITY * LOAD_PERCENTAGE / 100;

using HugeVecT = std::vector<size_t, huge_page_allocator<size_t>>;
using CuckooTableT = cuckoo_set::cuckoo_set<CRCHash<uint64_t>, huge_page_allocator<cuckoo_set::Bucket>>;

void run_test(const HugeVecT& read_idxs) {
  CuckooTableT table(CAPACITY);

  // do insertions
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    table.insert(i);
  }
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    auto it = table.find(i);
    assert(!it.is_null());
  }
  assert(table.size() == NUM_KEYS);

  std::array<cuckoo_set::cuckoo_worker<CRCHash<uint64_t>, huge_page_allocator<cuckoo_set::Bucket>>, 2> workers;
  workers[0].start(&table);
  workers[1].start(&table);

  // do lookups and measure throughput
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  using cuckoo_set::MAX_LOOKUP_BATCH_SZ;
  std::array<typename CuckooTableT::iterator, MAX_LOOKUP_BATCH_SZ> results{};
  for (size_t i = 0; i < NUM_REQUESTS; i += MAX_LOOKUP_BATCH_SZ) {
    workers[(i / MAX_LOOKUP_BATCH_SZ) % 2].queue(
      &read_idxs[i], MAX_LOOKUP_BATCH_SZ, results.data()
    );
  }

  for (auto &worker : workers) {
    worker.stop();
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
          .count();
  double throughput = static_cast<double>(NUM_REQUESTS) / (elapsed_us / 1e6);

  std::cout << "cuckoo_set lookup throughput: " << throughput << std::endl;

  // do deletions
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    auto it = table.find(i);
    table.erase(it);
  }
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    auto it = table.find(i);
    assert(it.is_null());
  }
  assert(table.size() == 0);
}

int main() {
  // generate random lookups
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> distrib(
      1, CAPACITY * LOAD_PERCENTAGE / HIT_PERCENTAGE);
  HugeVecT read_idxs(NUM_REQUESTS);
  for (size_t i = 0; i < NUM_REQUESTS; ++i) {
    read_idxs[i] = distrib(gen);
  }

  run_test(read_idxs);
}
