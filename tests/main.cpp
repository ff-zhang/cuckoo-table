#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "cuckoo_set.hpp"
#include "hash.hpp"
#include "huge_page_allocator.hpp"

constexpr size_t CAPACITY = 128 * 1024 * 1024;
constexpr size_t LOAD_PERCENTAGE = 80;
constexpr size_t HIT_PERCENTAGE = 80;
constexpr size_t NUM_REQUESTS = 100000000;
constexpr size_t NUM_KEYS = CAPACITY * LOAD_PERCENTAGE / 100;

constexpr size_t NUM_WORKERS = 2;

using HugeVecT = std::vector<size_t, huge_page_allocator<size_t>>;
using CuckooTableT = cuckoo_set::cuckoo_set<CRCHash<uint64_t>, huge_page_allocator<cuckoo_set::Bucket>>;

void run_test(const HugeVecT& read_idxs) {
  using cuckoo_set::MAX_LOOKUP_BATCH_SZ;

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

  std::array<size_t, NUM_WORKERS + 1> slices{0};
  for (size_t i = 1; i < NUM_WORKERS; ++i) {
    const size_t slice_ = i * NUM_REQUESTS / NUM_WORKERS;
    slices[i] = slice_ - slice_ % MAX_LOOKUP_BATCH_SZ;
  }
  slices[NUM_WORKERS] = NUM_REQUESTS;

  // do lookups and measure throughput
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  std::array<typename CuckooTableT::iterator, MAX_LOOKUP_BATCH_SZ> results{};
  auto worker = [&](const size_t start, const size_t end) {
    for (size_t i = start; i < end; i += MAX_LOOKUP_BATCH_SZ) {
      table.find_batched(&read_idxs[i], MAX_LOOKUP_BATCH_SZ, results.data());
    }
  };

  std::array<std::thread, NUM_WORKERS> workers;
  for (size_t i = 0 ; i < NUM_WORKERS; ++i) {
    workers[i] = std::thread(worker, slices[i], slices[i + 1]);
  }

  for (auto& t : workers) {
    t.join();
  }

  std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
  auto elapsed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - begin)
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
