#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define ALIGN_2MIB (2ULL * 1024ULL * 1024ULL)

/* Global sink to prevent dead-code elimination of the accumulator. */
static volatile uint64_t g_sink = 0;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --bytes N --passes P [--cpu C] [--warmup W] [--wait-signal]\n"
            "  --bytes N       Buffer size in bytes (positive multiple of 8)\n"
            "  --passes P      Number of timed passes (>= 1)\n"
            "  --cpu C         Pin process to CPU C\n"
            "  --warmup W      Number of warm-up passes (default: 1)\n"
            "  --wait-signal   Print {\"ready\": <pid>} and wait for SIGUSR1\n",
            prog);
}

static int parse_u64(const char *s, uint64_t *out) {
    if (s == NULL || *s == '\0' || *s == '-') {
        return -1;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long val = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return -1;
    }
    *out = (uint64_t)val;
    return 0;
}

static int parse_int_nonneg(const char *s, int *out) {
    if (s == NULL || *s == '\0' || *s == '-') {
        return -1;
    }
    errno = 0;
    char *end = NULL;
    long val = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || val < 0 || val > INT_MAX) {
        return -1;
    }
    *out = (int)val;
    return 0;
}

__attribute__((noinline))
static uint64_t read_pass(const uint64_t *words, size_t n_words) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n_words; i++) {
        acc += words[i];
    }
    return acc;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

static uint64_t timespec_diff_ns(const struct timespec *start,
                                 const struct timespec *end) {
    int64_t sec = (int64_t)end->tv_sec - (int64_t)start->tv_sec;
    int64_t nsec = (int64_t)end->tv_nsec - (int64_t)start->tv_nsec;
    int64_t total = sec * 1000000000LL + nsec;
    return total > 0 ? (uint64_t)total : 0ULL;
}

int main(int argc, char **argv) {
    uint64_t bytes = 0;
    int has_bytes = 0;
    uint64_t passes_u64 = 0;
    int has_passes = 0;
    int cpu = -1;
    int has_cpu = 0;
    int warmup = 1;
    int wait_signal = 0;

    static struct option long_opts[] = {
        {"bytes", required_argument, 0, 'b'},
        {"passes", required_argument, 0, 'p'},
        {"cpu", required_argument, 0, 'c'},
        {"warmup", required_argument, 0, 'w'},
        {"wait-signal", no_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    opterr = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'b':
                if (parse_u64(optarg, &bytes) != 0) {
                    print_usage(argv[0]);
                    return 2;
                }
                has_bytes = 1;
                break;
            case 'p':
                if (parse_u64(optarg, &passes_u64) != 0) {
                    print_usage(argv[0]);
                    return 2;
                }
                has_passes = 1;
                break;
            case 'c':
                if (parse_int_nonneg(optarg, &cpu) != 0) {
                    print_usage(argv[0]);
                    return 2;
                }
                has_cpu = 1;
                break;
            case 'w':
                if (parse_int_nonneg(optarg, &warmup) != 0) {
                    print_usage(argv[0]);
                    return 2;
                }
                break;
            case 's':
                wait_signal = 1;
                break;
            default:
                print_usage(argv[0]);
                return 2;
        }
    }

    if (optind < argc || !has_bytes || !has_passes) {
        print_usage(argv[0]);
        return 2;
    }
    if (bytes == 0 || (bytes % 8ULL) != 0 || passes_u64 == 0 ||
        passes_u64 > (uint64_t)SIZE_MAX / sizeof(uint64_t) ||
        bytes > (uint64_t)SIZE_MAX) {
        print_usage(argv[0]);
        return 2;
    }

    size_t n_bytes = (size_t)bytes;
    size_t n_words = n_bytes / 8;
    size_t passes = (size_t)passes_u64;

    if (has_cpu) {
        if (cpu >= CPU_SETSIZE) {
            fprintf(stderr, "CPU index %d exceeds CPU_SETSIZE (%d)\n", cpu, CPU_SETSIZE);
            return 1;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
            perror("sched_setaffinity");
            return 1;
        }
    }

    void *buf = NULL;
    int rc = posix_memalign(&buf, (size_t)ALIGN_2MIB, n_bytes);
    if (rc != 0 || buf == NULL) {
        fprintf(stderr, "posix_memalign failed: %s\n", strerror(rc));
        return 1;
    }

    (void)madvise(buf, n_bytes, MADV_HUGEPAGE);

    uint8_t *byte_buf = (uint8_t *)buf;
    for (size_t i = 0; i < n_bytes; i++) {
        byte_buf[i] = (uint8_t)(((uint64_t)i * 131ULL + 7ULL) & 0xffULL);
    }

    const uint64_t *words = (const uint64_t *)buf;
    uint64_t total_acc = 0;

    for (int w = 0; w < warmup; w++) {
        total_acc += read_pass(words, n_words);
        __asm__ volatile("" : : "r"(total_acc) : "memory");
    }

    if (wait_signal) {
        sigset_t sigset;
        sigemptyset(&sigset);
        sigaddset(&sigset, SIGUSR1);
        if (sigprocmask(SIG_BLOCK, &sigset, NULL) != 0) {
            perror("sigprocmask");
            free(buf);
            return 1;
        }
        printf("{\"ready\": %ld}\n", (long)getpid());
        fflush(stdout);
        int sig = 0;
        if (sigwait(&sigset, &sig) != 0) {
            perror("sigwait");
            free(buf);
            return 1;
        }
    }

    uint64_t *pass_ns = malloc(passes * sizeof(uint64_t));
    if (pass_ns == NULL) {
        perror("malloc");
        free(buf);
        return 1;
    }

    uint64_t checksum = 0;
    for (size_t p = 0; p < passes; p++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint64_t pass_sum = read_pass(words, n_words);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        total_acc += pass_sum;
        __asm__ volatile("" : : "r"(total_acc) : "memory");
        if (p == 0) {
            checksum = pass_sum;
        }

        uint64_t ns = timespec_diff_ns(&t0, &t1);
        pass_ns[p] = ns;
        double gbps = ns > 0 ? (double)n_bytes / (double)ns : 0.0;

        printf("{\"pass\": %zu, \"bytes\": %zu, \"ns\": %" PRIu64 ", \"gbps\": %.4f}\n",
               p, n_bytes, ns, gbps);
    }

    g_sink = total_acc;

    qsort(pass_ns, passes, sizeof(uint64_t), cmp_u64);
    uint64_t median_ns;
    if (passes % 2 == 1) {
        median_ns = pass_ns[passes / 2];
    } else {
        median_ns = (pass_ns[passes / 2 - 1] + pass_ns[passes / 2]) / 2;
    }

    printf("{\"summary\": true, \"bytes\": %zu, \"passes\": %zu, \"median_ns\": %" PRIu64 ", \"checksum\": %" PRIu64 "}\n",
           n_bytes, passes, median_ns, checksum);
    fflush(stdout);

    free(pass_ns);
    free(buf);
    return 0;
}
