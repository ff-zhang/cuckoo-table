# cuckoo-table
A bucketized cuckoo hash table implementation in C++, with support for SIMD (aarch64) and batched lookups.

## Setup
Only supports aarch64 machines for now.

Huge pages should be enabled to run the test:
```
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages
```

## Run Test
```
g++ test/main.cpp -std=c++17 -march=native -O2
./a.out
```

## Results
Lookup throughput on a Apple M3 Pro, running Ubuntu ARM.
```
cuckoo_set lookup throughput: 6.62527e+07
```
