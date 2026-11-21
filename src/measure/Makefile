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

vocab_coverage: vocab_coverage.cpp
	g++ -std=c++17 -O2 -Wall -Wno-deprecated-declarations -I../../third_party/llama.cpp/include -I../../third_party/llama.cpp/ggml/include vocab_coverage.cpp -L../../third_party/llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(abspath ../../third_party/llama.cpp/build/bin) -o vocab_coverage

clean:
	rm -f stream_bench pcie_bench vocab_coverage
