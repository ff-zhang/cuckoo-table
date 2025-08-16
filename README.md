# cuckoo-table
A bucketized cuckoo hash table implementation in C++ with support for SIMD (`aarch64`) and batched lookups.

## Setup
This only supports `aarch64` machines for now.

Huge pages should be enabled with the following command when running the test:
```
echo 2048 | sudo tee /proc/sys/vm/nr_hugepages
```

## Run Test
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```
