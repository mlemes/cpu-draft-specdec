#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_signal_received = 0;

void handle_signal(int sig) {
    g_signal_received = sig;
}

void install_signal_handlers() {
    struct sigaction sa {};
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

// ---------------------------------------------------------------------------
// Self-contained SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        state_[0] = 0x6a09e667U;
        state_[1] = 0xbb67ae85U;
        state_[2] = 0x3c6ef372U;
        state_[3] = 0xa54ff53aU;
        state_[4] = 0x510e527fU;
        state_[5] = 0x9b05688cU;
        state_[6] = 0x1f83d9abU;
        state_[7] = 0x5be0cd19U;
        bitlen_ = 0;
        buf_len_ = 0;
    }

    void update(const void * data, size_t len) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < len; ++i) {
            buf_[buf_len_++] = bytes[i];
            if (buf_len_ == 64) {
                transform(buf_);
                bitlen_ += 512;
                buf_len_ = 0;
            }
        }
    }

    std::string final_hex() {
        size_t i = buf_len_;
        if (buf_len_ < 56) {
            buf_[i++] = 0x80;
            while (i < 56) {
                buf_[i++] = 0x00;
            }
        } else {
            buf_[i++] = 0x80;
            while (i < 64) {
                buf_[i++] = 0x00;
            }
            transform(buf_);
            std::memset(buf_, 0, 56);
        }

        bitlen_ += static_cast<uint64_t>(buf_len_) * 8ULL;
        for (int j = 0; j < 8; ++j) {
            buf_[63 - j] = static_cast<uint8_t>((bitlen_ >> (j * 8)) & 0xFFU);
        }
        transform(buf_);

        static const char hex_digits[] = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int s = 0; s < 8; ++s) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                out.push_back(hex_digits[(state_[s] >> shift) & 0xFU]);
            }
        }
        return out;
    }

private:
    static inline uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t block[64]) {
        static const uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
        };

        uint32_t m[64];
        for (int i = 0; i < 16; ++i) {
            m[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                   (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
            uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
            m[i] = m[i - 16] + s0 + m[i - 7] + s1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + S1 + ch + k[i] + m[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    uint32_t state_[8]{};
    uint64_t bitlen_ = 0;
    uint8_t buf_[64]{};
    size_t buf_len_ = 0;
};

bool sha256_file(const std::string & path, std::string & out_hex) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return false;
    }
    Sha256 ctx;
    std::vector<char> buf(1 << 20);  // 1 MiB streaming buffer
    while (ifs) {
        ifs.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = ifs.gcount();
        if (got > 0) {
            ctx.update(buf.data(), static_cast<size_t>(got));
        }
    }
    if (ifs.bad()) {
        return false;
    }
    out_hex = ctx.final_hex();
    return true;
}

// ---------------------------------------------------------------------------
// CLI options & argument parsing
// ---------------------------------------------------------------------------

struct Options {
    std::string model_path;
    std::string input_path;
    std::string out_dir;
    int32_t seq_len = 1024;
    int32_t k = 32;
    int64_t max_tokens = -1;
    int32_t shard_seqs = 1024;
    int32_t ngl = 99;
    int32_t threads = 6;
    bool parse_special = false;
    int64_t dump_full = 0;
};

void print_usage(const char * prog) {
    std::fprintf(
        stderr,
        "Usage: %s --model GGUF --input DOCS.jsonl --out DIR "
        "[--seq-len 1024] [--k 32] [--max-tokens N] [--shard-seqs 1024] "
        "[--ngl 99] [--threads 6] [--parse-special] [--dump-full N]\n",
        prog);
}

bool parse_i32(const std::string & s, int32_t & out) {
    if (s.empty()) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    long val = std::strtol(s.c_str(), &end, 10);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') {
        return false;
    }
    if (val < INT32_MIN || val > INT32_MAX) {
        return false;
    }
    out = static_cast<int32_t>(val);
    return true;
}

bool parse_i64(const std::string & s, int64_t & out) {
    if (s.empty()) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    long long val = std::strtoll(s.c_str(), &end, 10);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int64_t>(val);
    return true;
}

bool parse_args(int argc, char ** argv, Options & opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--model") {
            const char * v = need_val("--model");
            if (!v || *v == '\0') return false;
            opts.model_path = v;
        } else if (arg == "--input") {
            const char * v = need_val("--input");
            if (!v || *v == '\0') return false;
            opts.input_path = v;
        } else if (arg == "--out") {
            const char * v = need_val("--out");
            if (!v || *v == '\0') return false;
            opts.out_dir = v;
        } else if (arg == "--seq-len") {
            const char * v = need_val("--seq-len");
            if (!v || !parse_i32(v, opts.seq_len) || opts.seq_len < 2) {
                std::fprintf(stderr, "Error: invalid --seq-len (must be integer >= 2)\n");
                return false;
            }
        } else if (arg == "--k") {
            const char * v = need_val("--k");
            if (!v || !parse_i32(v, opts.k) || opts.k < 1) {
                std::fprintf(stderr, "Error: invalid --k (must be integer >= 1)\n");
                return false;
            }
        } else if (arg == "--max-tokens") {
            const char * v = need_val("--max-tokens");
            if (!v || !parse_i64(v, opts.max_tokens) || opts.max_tokens < 0) {
                std::fprintf(stderr, "Error: invalid --max-tokens (must be integer >= 0)\n");
                return false;
            }
        } else if (arg == "--shard-seqs") {
            const char * v = need_val("--shard-seqs");
            if (!v || !parse_i32(v, opts.shard_seqs) || opts.shard_seqs < 1) {
                std::fprintf(stderr, "Error: invalid --shard-seqs (must be integer >= 1)\n");
                return false;
            }
        } else if (arg == "--ngl") {
            const char * v = need_val("--ngl");
            if (!v || !parse_i32(v, opts.ngl) || opts.ngl < 0) {
                std::fprintf(stderr, "Error: invalid --ngl (must be integer >= 0)\n");
                return false;
            }
        } else if (arg == "--threads") {
            const char * v = need_val("--threads");
            if (!v || !parse_i32(v, opts.threads) || opts.threads < 1) {
                std::fprintf(stderr, "Error: invalid --threads (must be integer >= 1)\n");
                return false;
            }
        } else if (arg == "--parse-special") {
            opts.parse_special = true;
        } else if (arg == "--dump-full") {
            const char * v = need_val("--dump-full");
            if (!v || !parse_i64(v, opts.dump_full) || opts.dump_full < 0) {
                std::fprintf(stderr, "Error: invalid --dump-full (must be integer >= 0)\n");
                return false;
            }
        } else {
            std::fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (opts.model_path.empty() || opts.input_path.empty() || opts.out_dir.empty()) {
        std::fprintf(stderr, "Error: --model, --input, and --out are required\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// JSON string / JSONL / index.json helpers (reused from vocab_coverage.cpp)
// ---------------------------------------------------------------------------

void append_utf8(uint32_t cp, std::string & out) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0x10FFFF) {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        append_utf8(0xFFFD, out);
    }
}

bool parse_hex4(const std::string & s, size_t pos, uint32_t & out) {
    if (pos + 4 > s.size()) {
        return false;
    }
    uint32_t val = 0;
    for (size_t i = 0; i < 4; ++i) {
        unsigned char c = static_cast<unsigned char>(s[pos + i]);
        val <<= 4;
        if (c >= '0' && c <= '9') {
            val |= (c - '0');
        } else if (c >= 'a' && c <= 'f') {
            val |= (c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            val |= (c - 'A' + 10);
        } else {
            return false;
        }
    }
    out = val;
    return true;
}

void skip_ws(const std::string & s, size_t & pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
}

bool parse_json_string(const std::string & s, size_t & pos, std::string & out) {
    if (pos >= s.size() || s[pos] != '"') {
        return false;
    }
    ++pos;
    out.clear();
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"') {
            return true;
        }
        if (c == '\\') {
            if (pos >= s.size()) {
                return false;
            }
            char esc = s[pos++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parse_hex4(s, pos, cp)) {
                        return false;
                    }
                    pos += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos + 6 <= s.size() && s[pos] == '\\' && s[pos + 1] == 'u') {
                            uint32_t low = 0;
                            if (parse_hex4(s, pos + 2, low) && low >= 0xDC00 && low <= 0xDFFF) {
                                pos += 6;
                                uint32_t full = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                                append_utf8(full, out);
                                break;
                            }
                        }
                        append_utf8(0xFFFD, out);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        append_utf8(0xFFFD, out);
                    } else {
                        append_utf8(cp, out);
                    }
                    break;
                }
                default:
                    return false;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

bool skip_json_value(const std::string & s, size_t & pos) {
    skip_ws(s, pos);
    if (pos >= s.size()) {
        return false;
    }
    char c = s[pos];
    if (c == '"') {
        std::string dummy;
        return parse_json_string(s, pos, dummy);
    }
    if (c == '{' || c == '[') {
        int depth = 0;
        while (pos < s.size()) {
            char ch = s[pos];
            if (ch == '"') {
                std::string dummy;
                if (!parse_json_string(s, pos, dummy)) {
                    return false;
                }
                continue;
            }
            if (ch == '{' || ch == '[') {
                ++depth;
            } else if (ch == '}' || ch == ']') {
                --depth;
                if (depth == 0) {
                    ++pos;
                    return true;
                }
            }
            ++pos;
        }
        return false;
    }
    size_t start = pos;
    while (pos < s.size()) {
        char ch = s[pos];
        if (ch == ',' || ch == '}' || ch == ']' || std::isspace(static_cast<unsigned char>(ch))) {
            return pos > start;
        }
        ++pos;
    }
    return pos > start;
}

bool parse_jsonl_text_row(const std::string & line, std::string & text) {
    size_t pos = 0;
    skip_ws(line, pos);
    if (pos >= line.size() || line[pos] != '{') {
        return false;
    }
    ++pos;
    bool has_text = false;
    text.clear();

    skip_ws(line, pos);
    if (pos < line.size() && line[pos] == '}') {
        return false;
    }

    while (pos < line.size()) {
        skip_ws(line, pos);
        std::string key;
        if (!parse_json_string(line, pos, key)) {
            return false;
        }
        skip_ws(line, pos);
        if (pos >= line.size() || line[pos] != ':') {
            return false;
        }
        ++pos;
        skip_ws(line, pos);
        if (key == "text") {
            if (!parse_json_string(line, pos, text)) {
                return false;
            }
            has_text = true;
        } else {
            if (!skip_json_value(line, pos)) {
                return false;
            }
        }
        skip_ws(line, pos);
        if (pos >= line.size()) {
            return false;
        }
        if (line[pos] == ',') {
            ++pos;
            continue;
        }
        if (line[pos] == '}') {
            ++pos;
            skip_ws(line, pos);
            return has_text && pos == line.size();
        }
        return false;
    }
    return false;
}

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

// ---------------------------------------------------------------------------
// Index manifest structures & resume parser
// ---------------------------------------------------------------------------

struct ShardInfo {
    std::string name;
    int32_t n_seq = 0;
    std::string sha_tokens;
    std::string sha_ids;
    std::string sha_logp;
};

struct IndexData {
    std::string format;
    std::string model;
    std::string model_sha256;
    std::string input;
    std::string input_sha256;
    int32_t seq_len = 0;
    int32_t k = 0;
    int32_t n_vocab = 0;
    int32_t eos_id = 0;
    bool parse_special = false;
    std::vector<ShardInfo> shards;
    int64_t n_seq = 0;
    int64_t n_tokens = 0;
    bool complete = false;
    double elapsed_s = 0.0;
    double tokens_per_s = 0.0;
};

bool parse_json_int64(const std::string & s, size_t & pos, int64_t & out) {
    skip_ws(s, pos);
    if (pos >= s.size()) return false;
    size_t start = pos;
    if (s[pos] == '-') ++pos;
    if (pos >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos]))) return false;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
    return parse_i64(s.substr(start, pos - start), out);
}

bool parse_json_bool(const std::string & s, size_t & pos, bool & out) {
    skip_ws(s, pos);
    if (s.compare(pos, 4, "true") == 0) {
        pos += 4;
        out = true;
        return true;
    }
    if (s.compare(pos, 5, "false") == 0) {
        pos += 5;
        out = false;
        return true;
    }
    return false;
}

bool parse_shard_sha256_obj(const std::string & s, size_t & pos, ShardInfo & info) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') return false;
    ++pos;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') {
        ++pos;
        return false;
    }
    while (pos < s.size()) {
        skip_ws(s, pos);
        std::string key;
        if (!parse_json_string(s, pos, key)) return false;
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        ++pos;
        skip_ws(s, pos);
        if (key == "tokens") {
            if (!parse_json_string(s, pos, info.sha_tokens)) return false;
        } else if (key == "ids") {
            if (!parse_json_string(s, pos, info.sha_ids)) return false;
        } else if (key == "logp") {
            if (!parse_json_string(s, pos, info.sha_logp)) return false;
        } else {
            if (!skip_json_value(s, pos)) return false;
        }
        skip_ws(s, pos);
        if (pos >= s.size()) return false;
        if (s[pos] == ',') {
            ++pos;
            continue;
        }
        if (s[pos] == '}') {
            ++pos;
            return !info.sha_tokens.empty() && !info.sha_ids.empty() && !info.sha_logp.empty();
        }
        return false;
    }
    return false;
}

bool parse_shard_obj(const std::string & s, size_t & pos, ShardInfo & info) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') return false;
    ++pos;
    bool has_name = false;
    bool has_n_seq = false;
    bool has_sha = false;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') return false;
    while (pos < s.size()) {
        skip_ws(s, pos);
        std::string key;
        if (!parse_json_string(s, pos, key)) return false;
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        ++pos;
        skip_ws(s, pos);
        if (key == "name") {
            if (!parse_json_string(s, pos, info.name)) return false;
            has_name = true;
        } else if (key == "n_seq") {
            int64_t val = 0;
            if (!parse_json_int64(s, pos, val) || val <= 0 || val > INT32_MAX) return false;
            info.n_seq = static_cast<int32_t>(val);
            has_n_seq = true;
        } else if (key == "sha256") {
            if (!parse_shard_sha256_obj(s, pos, info)) return false;
            has_sha = true;
        } else {
            if (!skip_json_value(s, pos)) return false;
        }
        skip_ws(s, pos);
        if (pos >= s.size()) return false;
        if (s[pos] == ',') {
            ++pos;
            continue;
        }
        if (s[pos] == '}') {
            ++pos;
            return has_name && has_n_seq && has_sha;
        }
        return false;
    }
    return false;
}

bool parse_shards_array(const std::string & s, size_t & pos, std::vector<ShardInfo> & shards) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '[') return false;
    ++pos;
    shards.clear();
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') {
        ++pos;
        return true;
    }
    while (pos < s.size()) {
        ShardInfo info;
        if (!parse_shard_obj(s, pos, info)) return false;
        shards.push_back(std::move(info));
        skip_ws(s, pos);
        if (pos >= s.size()) return false;
        if (s[pos] == ',') {
            ++pos;
            continue;
        }
        if (s[pos] == ']') {
            ++pos;
            return true;
        }
        return false;
    }
    return false;
}

bool parse_index_json(const std::string & path, IndexData & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::ostringstream oss;
    oss << ifs.rdbuf();
    if (ifs.bad()) return false;
    std::string s = oss.str();

    size_t pos = 0;
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '{') return false;
    ++pos;

    bool has_model_sha = false;
    bool has_input_sha = false;
    bool has_seq_len = false;
    bool has_k = false;
    bool has_parse_special = false;
    bool has_shards = false;

    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') return false;

    while (pos < s.size()) {
        skip_ws(s, pos);
        std::string key;
        if (!parse_json_string(s, pos, key)) return false;
        skip_ws(s, pos);
        if (pos >= s.size() || s[pos] != ':') return false;
        ++pos;
        skip_ws(s, pos);

        if (key == "format") {
            if (!parse_json_string(s, pos, out.format)) return false;
        } else if (key == "model") {
            if (!parse_json_string(s, pos, out.model)) return false;
        } else if (key == "model_sha256") {
            if (!parse_json_string(s, pos, out.model_sha256)) return false;
            has_model_sha = true;
        } else if (key == "input") {
            if (!parse_json_string(s, pos, out.input)) return false;
        } else if (key == "input_sha256") {
            if (!parse_json_string(s, pos, out.input_sha256)) return false;
            has_input_sha = true;
        } else if (key == "seq_len") {
            int64_t val = 0;
            if (!parse_json_int64(s, pos, val) || val < 2 || val > INT32_MAX) return false;
            out.seq_len = static_cast<int32_t>(val);
            has_seq_len = true;
        } else if (key == "k") {
            int64_t val = 0;
            if (!parse_json_int64(s, pos, val) || val < 1 || val > INT32_MAX) return false;
            out.k = static_cast<int32_t>(val);
            has_k = true;
        } else if (key == "parse_special") {
            if (!parse_json_bool(s, pos, out.parse_special)) return false;
            has_parse_special = true;
        } else if (key == "shards") {
            if (!parse_shards_array(s, pos, out.shards)) return false;
            has_shards = true;
        } else if (key == "complete") {
            if (!parse_json_bool(s, pos, out.complete)) return false;
        } else {
            if (!skip_json_value(s, pos)) return false;
        }

        skip_ws(s, pos);
        if (pos >= s.size()) return false;
        if (s[pos] == ',') {
            ++pos;
            continue;
        }
        if (s[pos] == '}') {
            ++pos;
            skip_ws(s, pos);
            return pos == s.size() && has_model_sha && has_input_sha &&
                   has_seq_len && has_k && has_parse_special && has_shards;
        }
        return false;
    }
    return false;
}

bool write_index_atomic(const fs::path & out_dir, const IndexData & idx) {
    fs::path tmp_path = out_dir / "index.json.tmp";
    fs::path final_path = out_dir / "index.json";

    std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"format\": " << json_escape(idx.format) << ",\n";
    ofs << "  \"model\": " << json_escape(idx.model) << ",\n";
    ofs << "  \"model_sha256\": " << json_escape(idx.model_sha256) << ",\n";
    ofs << "  \"input\": " << json_escape(idx.input) << ",\n";
    ofs << "  \"input_sha256\": " << json_escape(idx.input_sha256) << ",\n";
    ofs << "  \"seq_len\": " << idx.seq_len << ",\n";
    ofs << "  \"k\": " << idx.k << ",\n";
    ofs << "  \"n_vocab\": " << idx.n_vocab << ",\n";
    ofs << "  \"eos_id\": " << idx.eos_id << ",\n";
    ofs << "  \"parse_special\": " << (idx.parse_special ? "true" : "false") << ",\n";
    ofs << "  \"shards\": [\n";
    for (size_t i = 0; i < idx.shards.size(); ++i) {
        const auto & s = idx.shards[i];
        ofs << "    {\"name\": " << json_escape(s.name)
            << ", \"n_seq\": " << s.n_seq
            << ", \"sha256\": {\"tokens\": " << json_escape(s.sha_tokens)
            << ", \"ids\": " << json_escape(s.sha_ids)
            << ", \"logp\": " << json_escape(s.sha_logp) << "}}"
            << (i + 1 < idx.shards.size() ? "," : "") << "\n";
    }
    ofs << "  ],\n";
    ofs << "  \"n_seq\": " << idx.n_seq << ",\n";
    ofs << "  \"n_tokens\": " << idx.n_tokens << ",\n";
    ofs << "  \"complete\": " << (idx.complete ? "true" : "false") << ",\n";
    ofs << "  \"elapsed_s\": " << idx.elapsed_s << ",\n";
    ofs << "  \"tokens_per_s\": " << idx.tokens_per_s << "\n";
    ofs << "}\n";

    ofs.flush();
    if (!ofs.good()) {
        return false;
    }
    ofs.close();

    if (std::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tokenization & binary shard writing
// ---------------------------------------------------------------------------

bool tokenize_document(const struct llama_vocab * vocab,
                       const std::string & text,
                       bool parse_special,
                       std::vector<llama_token> & out_tokens) {
    out_tokens.clear();
    if (text.empty()) {
        return true;
    }
    if (text.size() > static_cast<size_t>(INT32_MAX)) {
        return false;
    }
    int32_t text_len = static_cast<int32_t>(text.size());
    int32_t n = llama_tokenize(vocab, text.data(), text_len, nullptr, 0, false, parse_special);
    int32_t cap = (n < 0) ? -n : n;
    if (cap <= 0) {
        return n == 0;
    }
    out_tokens.resize(static_cast<size_t>(cap));
    int32_t n2 = llama_tokenize(vocab, text.data(), text_len, out_tokens.data(), cap, false, parse_special);
    if (n2 < 0) {
        cap = -n2;
        out_tokens.resize(static_cast<size_t>(cap));
        n2 = llama_tokenize(vocab, text.data(), text_len, out_tokens.data(), cap, false, parse_special);
    }
    if (n2 < 0) {
        return false;
    }
    out_tokens.resize(static_cast<size_t>(n2));
    return true;
}

template <typename T>
bool write_binary_and_hash(const fs::path & path,
                           const std::vector<T> & data,
                           std::string & out_sha256) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    size_t nbytes = data.size() * sizeof(T);
    const char * raw = reinterpret_cast<const char *>(data.data());
    if (nbytes > 0) {
        ofs.write(raw, static_cast<std::streamsize>(nbytes));
    }
    ofs.flush();
    if (!ofs.good()) {
        return false;
    }
    Sha256 sha;
    if (nbytes > 0) {
        sha.update(raw, nbytes);
    }
    out_sha256 = sha.final_hex();
    return true;
}

// ---------------------------------------------------------------------------
// Per-position log-softmax + top-k worker
// ---------------------------------------------------------------------------

void process_positions_range(const std::vector<const float *> & logit_ptrs,
                             int32_t t_start,
                             int32_t t_end,
                             int32_t n_vocab,
                             int32_t k,
                             uint32_t * out_ids,
                             uint16_t * out_logp) {
    std::vector<int32_t> idx(static_cast<size_t>(n_vocab));
    for (int32_t t = t_start; t < t_end; ++t) {
        const float * logits = logit_ptrs[static_cast<size_t>(t)];

        double max_logit = static_cast<double>(logits[0]);
        for (int32_t v = 1; v < n_vocab; ++v) {
            double val = static_cast<double>(logits[v]);
            if (val > max_logit) {
                max_logit = val;
            }
        }

        double sum_exp = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            sum_exp += std::exp(static_cast<double>(logits[v]) - max_logit);
        }
        double lse = max_logit + std::log(sum_exp);

        std::iota(idx.begin(), idx.end(), 0);
        auto cmp = [logits](int32_t a, int32_t b) {
            if (logits[a] != logits[b]) {
                return logits[a] > logits[b];
            }
            return a < b;
        };
        if (k < n_vocab) {
            std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);
        }
        std::sort(idx.begin(), idx.begin() + k, cmp);

        size_t base = static_cast<size_t>(t) * static_cast<size_t>(k);
        for (int32_t j = 0; j < k; ++j) {
            int32_t tok_id = idx[static_cast<size_t>(j)];
            float logp_f32 = static_cast<float>(static_cast<double>(logits[tok_id]) - lse);
            ggml_fp16_t logp_f16 = ggml_fp32_to_fp16(logp_f32);
            out_ids[base + static_cast<size_t>(j)] = static_cast<uint32_t>(tok_id);
            out_logp[base + static_cast<size_t>(j)] = static_cast<uint16_t>(logp_f16);
        }
    }
}

bool write_dump_full(const fs::path & out_dir,
                     const std::vector<const float *> & logit_ptrs,
                     int32_t n_positions,
                     int32_t n_vocab) {
    fs::path dump_path = out_dir / "full.logp.f32";
    std::ofstream ofs(dump_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    std::vector<float> row(static_cast<size_t>(n_vocab));
    for (int32_t t = 0; t < n_positions; ++t) {
        const float * logits = logit_ptrs[static_cast<size_t>(t)];
        double max_logit = static_cast<double>(logits[0]);
        for (int32_t v = 1; v < n_vocab; ++v) {
            double val = static_cast<double>(logits[v]);
            if (val > max_logit) {
                max_logit = val;
            }
        }
        double sum_exp = 0.0;
        for (int32_t v = 0; v < n_vocab; ++v) {
            sum_exp += std::exp(static_cast<double>(logits[v]) - max_logit);
        }
        double lse = max_logit + std::log(sum_exp);
        for (int32_t v = 0; v < n_vocab; ++v) {
            row[static_cast<size_t>(v)] = static_cast<float>(static_cast<double>(logits[v]) - lse);
        }
        ofs.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!ofs.good()) {
            return false;
        }
    }
    ofs.flush();
    return ofs.good();
}

}  // namespace

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 2;
    }

    install_signal_handlers();

    // Compute input and model SHA-256 hashes before/alongside setup.
    std::string model_sha256;
    if (!sha256_file(opts.model_path, model_sha256)) {
        std::fprintf(stderr, "Error: failed to read model file '%s'\n", opts.model_path.c_str());
        return 1;
    }
    std::string input_sha256;
    if (!sha256_file(opts.input_path, input_sha256)) {
        std::fprintf(stderr, "Error: failed to read input file '%s'\n", opts.input_path.c_str());
        return 1;
    }

    // Check output directory / resume state before loading model.
    fs::path out_dir(opts.out_dir);
    fs::path index_path = out_dir / "index.json";
    IndexData index_data;
    int32_t next_shard_idx = 0;
    int64_t skip_seqs = 0;

    std::error_code ec;
    bool out_exists = fs::exists(out_dir, ec);
    if (ec) {
        std::fprintf(stderr, "Error: failed to inspect output dir '%s': %s\n",
                     opts.out_dir.c_str(), ec.message().c_str());
        return 1;
    }

    if (out_exists) {
        if (!fs::is_directory(out_dir, ec)) {
            std::fprintf(stderr, "Error: output path '%s' exists and is not a directory\n",
                         opts.out_dir.c_str());
            return 1;
        }
        if (!fs::exists(index_path, ec)) {
            std::fprintf(stderr, "Error: output dir '%s' exists without index.json\n",
                         opts.out_dir.c_str());
            return 1;
        }
        if (!parse_index_json(index_path.string(), index_data)) {
            std::fprintf(stderr, "Error: failed to parse existing '%s'\n",
                         index_path.c_str());
            return 1;
        }
        if (index_data.model_sha256 != model_sha256 ||
            index_data.input_sha256 != input_sha256 ||
            index_data.seq_len != opts.seq_len ||
            index_data.k != opts.k ||
            index_data.parse_special != opts.parse_special) {
            std::fprintf(stderr, "Error: resume mismatch in '%s'\n", index_path.c_str());
            return 1;
        }
        next_shard_idx = static_cast<int32_t>(index_data.shards.size());
        for (const auto & s : index_data.shards) {
            skip_seqs += s.n_seq;
        }
    } else {
        if (!fs::create_directories(out_dir, ec)) {
            std::fprintf(stderr, "Error: failed to create output dir '%s': %s\n",
                         opts.out_dir.c_str(), ec.message().c_str());
            return 1;
        }
    }

    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opts.ngl;
    struct llama_model * model = llama_model_load_from_file(opts.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "Error: failed to load model '%s'\n", opts.model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        std::fprintf(stderr, "Error: invalid vocab size %d\n", n_vocab);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (opts.k > n_vocab) {
        std::fprintf(stderr, "Error: --k (%d) exceeds vocabulary size (%d)\n", opts.k, n_vocab);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    llama_token eos_tok = llama_vocab_eos(vocab);

    index_data.format = "teacher-topk-v1";
    index_data.model = opts.model_path;
    index_data.model_sha256 = model_sha256;
    index_data.input = opts.input_path;
    index_data.input_sha256 = input_sha256;
    index_data.seq_len = opts.seq_len;
    index_data.k = opts.k;
    index_data.n_vocab = n_vocab;
    index_data.eos_id = static_cast<int32_t>(eos_tok);
    index_data.parse_special = opts.parse_special;
    index_data.n_seq = skip_seqs;
    index_data.n_tokens = skip_seqs * static_cast<int64_t>(opts.seq_len);
    index_data.complete = false;
    index_data.elapsed_s = 0.0;
    index_data.tokens_per_s = 0.0;

    if (!write_index_atomic(out_dir, index_data)) {
        std::fprintf(stderr, "Error: failed to write '%s'\n", index_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(opts.seq_len);
    cparams.n_batch = static_cast<uint32_t>(opts.seq_len);
    cparams.n_ubatch = static_cast<uint32_t>(std::min<int32_t>(512, opts.seq_len));
    cparams.n_seq_max = 1;
    struct llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        std::fprintf(stderr, "Error: failed to create llama_context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    struct llama_batch batch = llama_batch_init(opts.seq_len, 0, 1);
    batch.n_tokens = opts.seq_len;

    std::ifstream input_ifs(opts.input_path);
    if (!input_ifs) {
        std::fprintf(stderr, "Error: failed to open input '%s'\n", opts.input_path.c_str());
        llama_batch_free(batch);
        llama_free(lctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int64_t max_seqs = (opts.max_tokens >= 0) ? (opts.max_tokens / opts.seq_len) : -1;
    const size_t seq_len_sz = static_cast<size_t>(opts.seq_len);
    const size_t k_sz = static_cast<size_t>(opts.k);

    std::vector<uint32_t> shard_tokens;
    std::vector<uint32_t> shard_ids;
    std::vector<uint16_t> shard_logp;
    shard_tokens.reserve(static_cast<size_t>(opts.shard_seqs) * seq_len_sz);
    shard_ids.reserve(static_cast<size_t>(opts.shard_seqs) * seq_len_sz * k_sz);
    shard_logp.reserve(static_cast<size_t>(opts.shard_seqs) * seq_len_sz * k_sz);
    int32_t shard_n_seq = 0;

    std::vector<uint32_t> pending_tokens;
    std::vector<llama_token> doc_tokens;
    std::vector<const float *> logit_ptrs(seq_len_sz);
    std::vector<uint32_t> seq_ids(seq_len_sz * k_sz);
    std::vector<uint16_t> seq_logp(seq_len_sz * k_sz);

    auto start_time = std::chrono::steady_clock::now();
    int64_t total_seqs_seen = 0;
    int64_t new_seqs_scored = 0;
    int exit_code = 0;

    auto flush_shard = [&]() -> bool {
        if (shard_n_seq <= 0) {
            return true;
        }
        char name_buf[32];
        std::snprintf(name_buf, sizeof(name_buf), "shard-%05d", next_shard_idx);
        std::string shard_name(name_buf);

        std::string sha_tokens, sha_ids, sha_logp;
        if (!write_binary_and_hash(out_dir / (shard_name + ".tokens.u32"), shard_tokens, sha_tokens) ||
            !write_binary_and_hash(out_dir / (shard_name + ".ids.u32"), shard_ids, sha_ids) ||
            !write_binary_and_hash(out_dir / (shard_name + ".logp.f16"), shard_logp, sha_logp)) {
            std::fprintf(stderr, "Error: failed to write shard '%s'\n", shard_name.c_str());
            return false;
        }

        ShardInfo info;
        info.name = shard_name;
        info.n_seq = shard_n_seq;
        info.sha_tokens = sha_tokens;
        info.sha_ids = sha_ids;
        info.sha_logp = sha_logp;
        index_data.shards.push_back(std::move(info));
        index_data.n_seq += shard_n_seq;
        index_data.n_tokens = index_data.n_seq * static_cast<int64_t>(opts.seq_len);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double tok_per_s = (elapsed > 0.0)
                               ? (static_cast<double>(new_seqs_scored * opts.seq_len) / elapsed)
                               : 0.0;
        index_data.elapsed_s = elapsed;
        index_data.tokens_per_s = tok_per_s;

        if (!write_index_atomic(out_dir, index_data)) {
            std::fprintf(stderr, "Error: failed to update '%s'\n", index_path.c_str());
            return false;
        }

        std::fprintf(stderr, "[%s] sequences=%lld tokens=%lld tokens/s=%.1f\n",
                     shard_name.c_str(),
                     static_cast<long long>(index_data.n_seq),
                     static_cast<long long>(index_data.n_tokens),
                     tok_per_s);

        shard_tokens.clear();
        shard_ids.clear();
        shard_logp.clear();
        shard_n_seq = 0;
        ++next_shard_idx;
        return true;
    };

    std::string line;
    int64_t line_no = 0;
    bool stop_all = false;

    while (!stop_all && std::getline(input_ifs, line)) {
        ++line_no;
        if (g_signal_received != 0) {
            stop_all = true;
            break;
        }

        bool all_ws = true;
        for (unsigned char c : line) {
            if (!std::isspace(c)) {
                all_ws = false;
                break;
            }
        }
        if (all_ws) {
            continue;
        }

        std::string text;
        if (!parse_jsonl_text_row(line, text)) {
            std::fprintf(stderr, "Error: malformed JSONL at line %lld of '%s'\n",
                         static_cast<long long>(line_no), opts.input_path.c_str());
            exit_code = 1;
            stop_all = true;
            break;
        }

        if (!tokenize_document(vocab, text, opts.parse_special, doc_tokens)) {
            std::fprintf(stderr, "Error: tokenization failed at line %lld of '%s'\n",
                         static_cast<long long>(line_no), opts.input_path.c_str());
            exit_code = 1;
            stop_all = true;
            break;
        }

        for (llama_token t : doc_tokens) {
            pending_tokens.push_back(static_cast<uint32_t>(t));
        }
        pending_tokens.push_back(static_cast<uint32_t>(eos_tok));

        size_t offset = 0;
        while (pending_tokens.size() - offset >= seq_len_sz) {
            if (max_seqs >= 0 && total_seqs_seen >= max_seqs) {
                stop_all = true;
                break;
            }
            if (g_signal_received != 0) {
                stop_all = true;
                break;
            }

            const uint32_t * seq_ptr = pending_tokens.data() + offset;
            offset += seq_len_sz;

            if (total_seqs_seen < skip_seqs) {
                ++total_seqs_seen;
                continue;
            }

            llama_memory_clear(llama_get_memory(lctx), true);
            for (int32_t i = 0; i < opts.seq_len; ++i) {
                batch.token[i] = static_cast<llama_token>(seq_ptr[i]);
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 1;
            }

            if (llama_decode(lctx, batch) != 0) {
                std::fprintf(stderr, "Error: llama_decode failed at sequence %lld\n",
                             static_cast<long long>(total_seqs_seen));
                exit_code = 1;
                stop_all = true;
                break;
            }

            for (int32_t t = 0; t < opts.seq_len; ++t) {
                const float * ptr = llama_get_logits_ith(lctx, t);
                if (!ptr) {
                    std::fprintf(stderr, "Error: llama_get_logits_ith(%d) returned null\n", t);
                    exit_code = 1;
                    stop_all = true;
                    break;
                }
                logit_ptrs[static_cast<size_t>(t)] = ptr;
            }
            if (stop_all) {
                break;
            }

            int32_t n_threads = std::min(opts.threads, opts.seq_len);
            if (n_threads <= 1) {
                process_positions_range(logit_ptrs, 0, opts.seq_len, n_vocab, opts.k,
                                        seq_ids.data(), seq_logp.data());
            } else {
                std::vector<std::thread> workers;
                workers.reserve(static_cast<size_t>(n_threads - 1));
                int32_t base_chunk = opts.seq_len / n_threads;
                int32_t rem = opts.seq_len % n_threads;
                int32_t cur = 0;
                for (int32_t w = 0; w < n_threads - 1; ++w) {
                    int32_t len = base_chunk + (w < rem ? 1 : 0);
                    workers.emplace_back(process_positions_range,
                                         std::cref(logit_ptrs),
                                         cur,
                                         cur + len,
                                         n_vocab,
                                         opts.k,
                                         seq_ids.data(),
                                         seq_logp.data());
                    cur += len;
                }
                process_positions_range(logit_ptrs, cur, opts.seq_len, n_vocab, opts.k,
                                        seq_ids.data(), seq_logp.data());
                for (auto & th : workers) {
                    th.join();
                }
            }

            if (total_seqs_seen == 0 && opts.dump_full > 0) {
                int32_t n_dump = static_cast<int32_t>(
                    std::min<int64_t>(opts.dump_full, opts.seq_len));
                if (!write_dump_full(out_dir, logit_ptrs, n_dump, n_vocab)) {
                    std::fprintf(stderr, "Error: failed to write full.logp.f32\n");
                    exit_code = 1;
                    stop_all = true;
                    break;
                }
            }

            shard_tokens.insert(shard_tokens.end(), seq_ptr, seq_ptr + seq_len_sz);
            shard_ids.insert(shard_ids.end(), seq_ids.begin(), seq_ids.end());
            shard_logp.insert(shard_logp.end(), seq_logp.begin(), seq_logp.end());
            ++shard_n_seq;
            ++total_seqs_seen;
            ++new_seqs_scored;

            if (shard_n_seq == opts.shard_seqs) {
                if (!flush_shard()) {
                    exit_code = 1;
                    stop_all = true;
                    break;
                }
            }

            if (g_signal_received != 0) {
                stop_all = true;
                break;
            }
        }

        if (offset > 0) {
            pending_tokens.erase(
                pending_tokens.begin(),
                pending_tokens.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }

    if (exit_code == 0 && input_ifs.bad()) {
        std::fprintf(stderr, "Error: I/O error reading '%s'\n", opts.input_path.c_str());
        exit_code = 1;
    }

    if (exit_code == 0) {
        if (g_signal_received != 0) {
            // Do not write a partial shard; leave complete: false.
            exit_code = 128 + static_cast<int>(g_signal_received);
        } else {
            if (shard_n_seq > 0) {
                if (!flush_shard()) {
                    exit_code = 1;
                }
            }
            if (exit_code == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double tok_per_s = (elapsed > 0.0)
                                       ? (static_cast<double>(new_seqs_scored * opts.seq_len) / elapsed)
                                       : 0.0;
                index_data.complete = true;
                index_data.elapsed_s = elapsed;
                index_data.tokens_per_s = tok_per_s;
                if (!write_index_atomic(out_dir, index_data)) {
                    std::fprintf(stderr, "Error: failed to write final '%s'\n", index_path.c_str());
                    exit_code = 1;
                }
            }
        }
    }

    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
