#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

__global__ void empty_kernel() {}

void print_usage(const char* prog) {
    std::fprintf(
        stderr,
        "Usage: %s [--sizes B1,B2,...] [--iters N] [--warmup W] [--device D]\n",
        prog);
}

bool parse_size_t(const char* s, std::size_t* out) {
    if (s == nullptr || *s == '\0' || *s == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long val = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    *out = static_cast<std::size_t>(val);
    return true;
}

bool parse_int(const char* s, int* out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    long val = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    if (val < -2147483648L || val > 2147483647L) {
        return false;
    }
    *out = static_cast<int>(val);
    return true;
}

bool parse_sizes(const char* arg, std::vector<std::size_t>* out) {
    if (arg == nullptr || *arg == '\0') {
        return false;
    }
    std::vector<std::size_t> parsed;
    const char* p = arg;
    while (true) {
        const char* comma = std::strchr(p, ',');
        std::string token = comma ? std::string(p, comma - p) : std::string(p);
        std::size_t sz = 0;
        if (!parse_size_t(token.c_str(), &sz) || sz == 0) {
            return false;
        }
        parsed.push_back(sz);
        if (!comma) {
            break;
        }
        p = comma + 1;
        if (*p == '\0') {
            return false;
        }
    }
    if (parsed.empty()) {
        return false;
    }
    *out = std::move(parsed);
    return true;
}

struct Stats {
    double median_us;
    double p95_us;
};

Stats compute_stats(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    std::size_t n = samples.size();
    double median = (n % 2 == 1)
                        ? samples[n / 2]
                        : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    std::size_t p95_idx = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n)));
    if (p95_idx > 0) {
        p95_idx -= 1;
    }
    if (p95_idx >= n) {
        p95_idx = n - 1;
    }
    return {median, samples[p95_idx]};
}

}  // namespace

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t err__ = (call);                                                    \
        if (err__ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(err__));                                   \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes = {1024, 4096, 65536, 262144, 1048576, 3145728};
    int iters = 200;
    int warmup = 20;
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--sizes") == 0) {
            if (i + 1 >= argc || !parse_sizes(argv[++i], &sizes)) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &iters) || iters <= 0) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &warmup) || warmup < 0) {
                print_usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--device") == 0) {
            if (i + 1 >= argc || !parse_int(argv[++i], &device) || device < 0) {
                print_usage(argv[0]);
                return 2;
            }
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    int device_count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(count_err));
        return 1;
    }
    if (device_count <= 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    if (device >= device_count) {
        std::fprintf(stderr, "CUDA device %d out of range (count=%d)\n", device, device_count);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(device));

    std::size_t max_size = *std::max_element(sizes.begin(), sizes.end());

    void* d_buf = nullptr;
    void* h_pinned = nullptr;
    void* h_pageable = std::malloc(max_size);
    if (h_pageable == nullptr) {
        std::fprintf(stderr, "Failed to allocate %zu bytes of pageable host memory\n", max_size);
        return 1;
    }

    CUDA_CHECK(cudaMalloc(&d_buf, max_size));
    CUDA_CHECK(cudaMallocHost(&h_pinned, max_size));

    std::memset(h_pageable, 0x5A, max_size);
    std::memset(h_pinned, 0xA5, max_size);
    CUDA_CHECK(cudaMemset(d_buf, 0, max_size));
    CUDA_CHECK(cudaDeviceSynchronize());

    struct TransferCase {
        const char* dir;
        cudaMemcpyKind kind_enum;
        const char* mem_kind;
        void* host_ptr;
    };

    const TransferCase cases[] = {
        {"h2d", cudaMemcpyHostToDevice, "pinned", h_pinned},
        {"h2d", cudaMemcpyHostToDevice, "pageable", h_pageable},
        {"d2h", cudaMemcpyDeviceToHost, "pinned", h_pinned},
        {"d2h", cudaMemcpyDeviceToHost, "pageable", h_pageable},
    };

    std::vector<double> samples(static_cast<std::size_t>(iters));

    for (std::size_t sz : sizes) {
        for (const auto& tc : cases) {
            void* dst = (tc.kind_enum == cudaMemcpyHostToDevice) ? d_buf : tc.host_ptr;
            const void* src = (tc.kind_enum == cudaMemcpyHostToDevice) ? tc.host_ptr : d_buf;

            for (int w = 0; w < warmup; ++w) {
                CUDA_CHECK(cudaMemcpy(dst, src, sz, tc.kind_enum));
            }
            for (int it = 0; it < iters; ++it) {
                auto t0 = std::chrono::steady_clock::now();
                CUDA_CHECK(cudaMemcpy(dst, src, sz, tc.kind_enum));
                auto t1 = std::chrono::steady_clock::now();
                samples[static_cast<std::size_t>(it)] =
                    std::chrono::duration<double, std::micro>(t1 - t0).count();
            }

            Stats st = compute_stats(samples);
            double gbps = (st.median_us > 0.0)
                              ? (static_cast<double>(sz) / (st.median_us * 1e3))
                              : 0.0;
            std::printf(
                "{\"kind\": \"%s\", \"dir\": \"%s\", \"bytes\": %zu, \"iters\": %d, "
                "\"median_us\": %.3f, \"p95_us\": %.3f, \"gbps\": %.6f}\n",
                tc.mem_kind, tc.dir, sz, iters, st.median_us, st.p95_us, gbps);
            std::fflush(stdout);
        }
    }

    for (int w = 0; w < warmup; ++w) {
        empty_kernel<<<1, 1>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        empty_kernel<<<1, 1>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        samples[static_cast<std::size_t>(it)] =
            std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    Stats launch_st = compute_stats(samples);
    std::printf(
        "{\"kind\": \"launch_sync\", \"iters\": %d, \"median_us\": %.3f, \"p95_us\": %.3f}\n",
        iters, launch_st.median_us, launch_st.p95_us);
    std::fflush(stdout);

    cudaFreeHost(h_pinned);
    cudaFree(d_buf);
    std::free(h_pageable);
    return 0;
}
