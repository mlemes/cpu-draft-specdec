#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_path;
    std::vector<std::string> freq_files;
    std::vector<std::string> eval_files;
    std::string out_path;
    std::vector<int32_t> sizes = {16384, 32768, 65536};
    int32_t ctx = 1024;
    int32_t ngl = 99;
    int64_t max_rows = -1;
    std::string ranking_out_path;
};

struct RowResult {
    std::string file;
    int64_t row = 0;
    int32_t n_positions = 0;
    std::vector<double> mass;
    std::vector<double> greedy_in;
    std::vector<double> actual_in;
};

void print_usage(const char * prog) {
    std::fprintf(
        stderr,
        "Usage: %s --model GGUF --freq-text F1[,F2...] --eval P1[,P2...] --out OUT.json "
        "[--sizes 16384,32768,65536] [--ctx 1024] [--ngl 99] [--max-rows N] [--ranking-out RANK.json]\n",
        prog);
}

bool split_nonempty(const std::string & s, char delim, std::vector<std::string> & out) {
    out.clear();
    if (s.empty()) {
        return false;
    }
    size_t start = 0;
    while (start <= s.size()) {
        size_t pos = s.find(delim, start);
        std::string item = (pos == std::string::npos) ? s.substr(start) : s.substr(start, pos - start);
        if (item.empty()) {
            return false;
        }
        out.push_back(item);
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return !out.empty();
}

bool parse_i32(const std::string & s, int32_t & out) {
    if (s.empty()) {
        return false;
    }
    char * end = nullptr;
    long val = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
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
    char * end = nullptr;
    long long val = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
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
        } else if (arg == "--freq-text") {
            const char * v = need_val("--freq-text");
            if (!v || !split_nonempty(v, ',', opts.freq_files)) {
                std::fprintf(stderr, "Error: invalid --freq-text\n");
                return false;
            }
        } else if (arg == "--eval") {
            const char * v = need_val("--eval");
            if (!v || !split_nonempty(v, ',', opts.eval_files)) {
                std::fprintf(stderr, "Error: invalid --eval\n");
                return false;
            }
        } else if (arg == "--out") {
            const char * v = need_val("--out");
            if (!v || *v == '\0') return false;
            opts.out_path = v;
        } else if (arg == "--sizes") {
            const char * v = need_val("--sizes");
            std::vector<std::string> parts;
            if (!v || !split_nonempty(v, ',', parts)) {
                std::fprintf(stderr, "Error: invalid --sizes\n");
                return false;
            }
            opts.sizes.clear();
            for (const auto & p : parts) {
                int32_t sz = 0;
                if (!parse_i32(p, sz) || sz <= 0) {
                    std::fprintf(stderr, "Error: invalid size '%s'\n", p.c_str());
                    return false;
                }
                opts.sizes.push_back(sz);
            }
        } else if (arg == "--ctx") {
            const char * v = need_val("--ctx");
            if (!v || !parse_i32(v, opts.ctx) || opts.ctx <= 0) {
                std::fprintf(stderr, "Error: invalid --ctx\n");
                return false;
            }
        } else if (arg == "--ngl") {
            const char * v = need_val("--ngl");
            if (!v || !parse_i32(v, opts.ngl) || opts.ngl < 0) {
                std::fprintf(stderr, "Error: invalid --ngl\n");
                return false;
            }
        } else if (arg == "--max-rows") {
            const char * v = need_val("--max-rows");
            if (!v || !parse_i64(v, opts.max_rows) || opts.max_rows < 0) {
                std::fprintf(stderr, "Error: invalid --max-rows\n");
                return false;
            }
        } else if (arg == "--ranking-out") {
            const char * v = need_val("--ranking-out");
            if (!v || *v == '\0') return false;
            opts.ranking_out_path = v;
        } else {
            std::fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (opts.model_path.empty() || opts.freq_files.empty() || opts.eval_files.empty() || opts.out_path.empty()) {
        std::fprintf(stderr, "Error: --model, --freq-text, --eval, and --out are required\n");
        return false;
    }
    return true;
}

bool read_entire_file(const std::string & path, std::string & content) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    if (ifs.bad()) {
        return false;
    }
    content = oss.str();
    return true;
}

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
    while (pos < s.size()) {
        char ch = s[pos];
        if (ch == ',' || ch == '}' || ch == ']' || std::isspace(static_cast<unsigned char>(ch))) {
            return true;
        }
        ++pos;
    }
    return true;
}

bool parse_jsonl_row(const std::string & line, std::string & prompt, std::string & heldout) {
    size_t pos = 0;
    skip_ws(line, pos);
    if (pos >= line.size() || line[pos] != '{') {
        return false;
    }
    ++pos;
    bool has_prompt = false;
    bool has_heldout = false;
    prompt.clear();
    heldout.clear();

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
        if (key == "prompt") {
            if (!parse_json_string(line, pos, prompt)) {
                return false;
            }
            has_prompt = true;
        } else if (key == "heldout") {
            if (!parse_json_string(line, pos, heldout)) {
                return false;
            }
            has_heldout = true;
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
            break;
        }
        return false;
    }
    return has_prompt && has_heldout;
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

std::vector<llama_token> tokenize_text(const struct llama_vocab * vocab,
                                       const std::string & text,
                                       bool add_special,
                                       bool parse_special) {
    int32_t text_len = static_cast<int32_t>(text.size());
    int32_t n = llama_tokenize(vocab, text.data(), text_len, nullptr, 0, add_special, parse_special);
    int32_t cap = (n < 0) ? -n : n;
    if (cap <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<size_t>(cap));
    int32_t n2 = llama_tokenize(vocab, text.data(), text_len, tokens.data(), cap, add_special, parse_special);
    if (n2 < 0) {
        cap = -n2;
        tokens.resize(static_cast<size_t>(cap));
        n2 = llama_tokenize(vocab, text.data(), text_len, tokens.data(), cap, add_special, parse_special);
    }
    if (n2 < 0) {
        return {};
    }
    tokens.resize(static_cast<size_t>(n2));
    return tokens;
}

}  // namespace

int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 2;
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
    bool add_bos = llama_vocab_get_add_bos(vocab);

    // 1. Frequency ranking
    std::vector<uint64_t> counts(static_cast<size_t>(n_vocab), 0);
    for (const auto & freq_path : opts.freq_files) {
        std::string content;
        if (!read_entire_file(freq_path, content)) {
            std::fprintf(stderr, "Error: failed to read freq file '%s'\n", freq_path.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        std::vector<llama_token> toks = tokenize_text(vocab, content, false, true);
        for (llama_token t : toks) {
            if (t >= 0 && t < n_vocab) {
                counts[static_cast<size_t>(t)] += 1;
            }
        }
    }

    std::vector<int32_t> ranked(static_cast<size_t>(n_vocab));
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&counts](int32_t a, int32_t b) {
        if (counts[static_cast<size_t>(a)] != counts[static_cast<size_t>(b)]) {
            return counts[static_cast<size_t>(a)] > counts[static_cast<size_t>(b)];
        }
        return a < b;
    });

    std::vector<int32_t> rank_of(static_cast<size_t>(n_vocab));
    for (int32_t r = 0; r < n_vocab; ++r) {
        rank_of[static_cast<size_t>(ranked[static_cast<size_t>(r)])] = r;
    }

    // Sort indices into opts.sizes so we can accumulate mass in one pass per position.
    std::vector<size_t> size_order(opts.sizes.size());
    std::iota(size_order.begin(), size_order.end(), 0);
    std::sort(size_order.begin(), size_order.end(), [&](size_t a, size_t b) {
        return opts.sizes[a] < opts.sizes[b];
    });
    int32_t max_size = opts.sizes[size_order.back()];

    if (!opts.ranking_out_path.empty()) {
        std::ofstream rofs(opts.ranking_out_path);
        if (!rofs) {
            std::fprintf(stderr, "Error: failed to open ranking output '%s'\n", opts.ranking_out_path.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        int32_t rank_count = std::min(max_size, n_vocab);
        rofs << "[";
        for (int32_t i = 0; i < rank_count; ++i) {
            if (i > 0) rofs << ",";
            rofs << ranked[static_cast<size_t>(i)];
        }
        rofs << "]\n";
        if (!rofs.good()) {
            std::fprintf(stderr, "Error: failed to write ranking output '%s'\n", opts.ranking_out_path.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    // 2-3. Evaluation
    std::vector<RowResult> results;
    std::vector<double> probs(static_cast<size_t>(n_vocab));

    for (const auto & eval_path : opts.eval_files) {
        std::ifstream eifs(eval_path);
        if (!eifs) {
            std::fprintf(stderr, "Error: failed to open eval file '%s'\n", eval_path.c_str());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }

        std::string line;
        int64_t row_idx = 0;
        while (std::getline(eifs, line)) {
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
            if (opts.max_rows >= 0 && row_idx >= opts.max_rows) {
                break;
            }

            std::string prompt;
            std::string heldout;
            if (!parse_jsonl_row(line, prompt, heldout)) {
                std::fprintf(stderr, "Error: invalid JSONL row %lld in '%s'\n",
                             static_cast<long long>(row_idx), eval_path.c_str());
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }

            std::string text = (!prompt.empty() && prompt.back() == '\n')
                                   ? (prompt + heldout)
                                   : (prompt + " " + heldout);
            std::vector<llama_token> tokens = tokenize_text(vocab, text, add_bos, true);
            if (static_cast<int32_t>(tokens.size()) > opts.ctx) {
                tokens.resize(static_cast<size_t>(opts.ctx));
            }

            RowResult rr;
            rr.file = eval_path;
            rr.row = row_idx;
            rr.mass.assign(opts.sizes.size(), 0.0);
            rr.greedy_in.assign(opts.sizes.size(), 0.0);
            rr.actual_in.assign(opts.sizes.size(), 0.0);

            int32_t n_tok = static_cast<int32_t>(tokens.size());
            if (n_tok >= 2) {
                int32_t n_pos = n_tok - 1;
                rr.n_positions = n_pos;

                struct llama_context_params cparams = llama_context_default_params();
                cparams.n_ctx = static_cast<uint32_t>(opts.ctx);
                cparams.n_batch = static_cast<uint32_t>(opts.ctx);
                cparams.n_ubatch = static_cast<uint32_t>(opts.ctx);
                struct llama_context * lctx = llama_init_from_model(model, cparams);
                if (!lctx) {
                    std::fprintf(stderr, "Error: failed to create llama_context\n");
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }

                struct llama_batch batch = llama_batch_init(n_tok, 0, 1);
                batch.n_tokens = n_tok;
                for (int32_t i = 0; i < n_tok; ++i) {
                    batch.token[i] = tokens[static_cast<size_t>(i)];
                    batch.pos[i] = i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = 1;
                }

                if (llama_decode(lctx, batch) != 0) {
                    std::fprintf(stderr, "Error: llama_decode failed on row %lld of '%s'\n",
                                 static_cast<long long>(row_idx), eval_path.c_str());
                    llama_batch_free(batch);
                    llama_free(lctx);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }

                int32_t limit = std::min(max_size, n_vocab);
                for (int32_t t = 0; t < n_pos; ++t) {
                    const float * logits = llama_get_logits_ith(lctx, t);
                    if (!logits) {
                        std::fprintf(stderr, "Error: llama_get_logits_ith returned null\n");
                        llama_batch_free(batch);
                        llama_free(lctx);
                        llama_model_free(model);
                        llama_backend_free();
                        return 1;
                    }

                    double max_logit = static_cast<double>(logits[0]);
                    int32_t argmax_id = 0;
                    for (int32_t v = 1; v < n_vocab; ++v) {
                        double val = static_cast<double>(logits[v]);
                        if (val > max_logit) {
                            max_logit = val;
                            argmax_id = v;
                        }
                    }

                    double sum_exp = 0.0;
                    for (int32_t v = 0; v < n_vocab; ++v) {
                        double e = std::exp(static_cast<double>(logits[v]) - max_logit);
                        probs[static_cast<size_t>(v)] = e;
                        sum_exp += e;
                    }
                    double inv_sum = 1.0 / sum_exp;
                    for (int32_t v = 0; v < n_vocab; ++v) {
                        probs[static_cast<size_t>(v)] *= inv_sum;
                    }

                    double running_mass = 0.0;
                    size_t ord_idx = 0;
                    for (int32_t r = 0; r < limit; ++r) {
                        running_mass += probs[static_cast<size_t>(ranked[static_cast<size_t>(r)])];
                        while (ord_idx < size_order.size() &&
                               r + 1 == opts.sizes[size_order[ord_idx]]) {
                            rr.mass[size_order[ord_idx]] += running_mass;
                            ++ord_idx;
                        }
                    }
                    while (ord_idx < size_order.size()) {
                        rr.mass[size_order[ord_idx]] += running_mass;
                        ++ord_idx;
                    }

                    int32_t argmax_rank = rank_of[static_cast<size_t>(argmax_id)];
                    llama_token next_tok = tokens[static_cast<size_t>(t + 1)];
                    int32_t actual_rank = (next_tok >= 0 && next_tok < n_vocab)
                                              ? rank_of[static_cast<size_t>(next_tok)]
                                              : n_vocab;
                    for (size_t si = 0; si < opts.sizes.size(); ++si) {
                        int32_t k = opts.sizes[si];
                        if (argmax_rank < k) {
                            rr.greedy_in[si] += 1.0;
                        }
                        if (actual_rank < k) {
                            rr.actual_in[si] += 1.0;
                        }
                    }
                }

                double inv_pos = 1.0 / static_cast<double>(n_pos);
                for (size_t si = 0; si < opts.sizes.size(); ++si) {
                    rr.mass[si] *= inv_pos;
                    rr.greedy_in[si] *= inv_pos;
                    rr.actual_in[si] *= inv_pos;
                }

                llama_batch_free(batch);
                llama_free(lctx);
            }

            results.push_back(std::move(rr));
            ++row_idx;
        }
    }

    // 4. Write OUT.json
    std::ofstream ofs(opts.out_path);
    if (!ofs) {
        std::fprintf(stderr, "Error: failed to open output '%s'\n", opts.out_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    ofs << std::setprecision(10);
    ofs << "{\n";
    ofs << "  \"model\": " << json_escape(opts.model_path) << ",\n";
    ofs << "  \"sizes\": [";
    for (size_t i = 0; i < opts.sizes.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << opts.sizes[i];
    }
    ofs << "],\n";
    ofs << "  \"ctx\": " << opts.ctx << ",\n";
    ofs << "  \"freq_files\": [";
    for (size_t i = 0; i < opts.freq_files.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << json_escape(opts.freq_files[i]);
    }
    ofs << "],\n";
    ofs << "  \"rows\": [\n";
    for (size_t r = 0; r < results.size(); ++r) {
        const auto & rr = results[r];
        ofs << "    {\n";
        ofs << "      \"file\": " << json_escape(rr.file) << ",\n";
        ofs << "      \"row\": " << rr.row << ",\n";
        ofs << "      \"n_positions\": " << rr.n_positions << ",\n";

        auto write_metric_map = [&](const char * name, const std::vector<double> & vals, bool trailing_comma) {
            ofs << "      \"" << name << "\": {";
            for (size_t si = 0; si < opts.sizes.size(); ++si) {
                if (si > 0) ofs << ", ";
                ofs << "\"" << opts.sizes[si] << "\": " << vals[si];
            }
            ofs << "}" << (trailing_comma ? "," : "") << "\n";
        };

        write_metric_map("mass", rr.mass, true);
        write_metric_map("greedy_in", rr.greedy_in, true);
        write_metric_map("actual_in", rr.actual_in, false);
        ofs << "    }" << (r + 1 < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";

    if (!ofs.good()) {
        std::fprintf(stderr, "Error: failed to write output '%s'\n", opts.out_path.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
