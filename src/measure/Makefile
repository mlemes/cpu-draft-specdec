CC := gcc
CFLAGS := -O2 -std=c11 -D_GNU_SOURCE -Wall -Wextra
NVCC := nvcc
NVCCFLAGS := -ccbin /usr/bin/g++-13 -O2 -arch=sm_86

.PHONY: all clean

all: stream_bench

stream_bench: stream_bench.c
	$(CC) $(CFLAGS) -o $@ $<

pcie_bench: pcie_bench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f stream_bench pcie_bench
