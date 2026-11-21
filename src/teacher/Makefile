CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -ffast-math -Wall -Wno-deprecated-declarations -I../../third_party/llama.cpp/include -I../../third_party/llama.cpp/ggml/include
LDFLAGS := -L../../third_party/llama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(abspath ../../third_party/llama.cpp/build/bin) -pthread

.PHONY: all clean

all: teacher_score

teacher_score: teacher_score.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f teacher_score
