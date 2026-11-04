#include "../src/models/qwen4exp-pca.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Synthetic single-query QSA gather, score and top-512 selection. No model load.
class score_graph {
public:
    score_graph(ggml_backend_t backend, int pools, bool projected) : backend_(backend) {
        ctx_ = ggml_init({2 * 1024 * 1024, nullptr, true});
        if (!ctx_) { throw std::runtime_error("metadata allocation failed"); }
        auto * raw = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 256, pools);
        auto * query = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 128, 4);
        auto * ids = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, pools);
        auto * mask = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, pools, 1);
        auto * projection = projected ? ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 128, 64) : nullptr;
        auto * weights = projected ? nullptr : ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 4, 1);
        const int width = projected ? 64 : 128;
        auto * view = ggml_view_2d(ctx_, raw, width, pools, raw->nb[1], 128 * sizeof(float));
        auto * keys = ggml_reshape_3d(ctx_, ggml_get_rows(ctx_, view, ids), width, 1, pools);
        ggml_tensor * q = query;
        if (projected) {
            q = ggml_mul_mat(ctx_, projection, q);
            ggml_prec_set_acc(q, GGML_PREC_F32);
            ggml_prec_set_src(q, GGML_PREC_F16, 1);
        }
        q = ggml_reshape_3d(ctx_, q, width, 4, 1);
        auto * score = projected ? qwen4exp_pca_score(ctx_, q, keys, mask, 1.0f / std::sqrt(128.0f)) :
            ggml_lightning_indexer(ctx_, q, keys, weights, mask);
        output_ = ggml_top_k(ctx_, score, 512);
        graph_ = ggml_new_graph_custom(ctx_, 128, false);
        ggml_build_forward_expand(graph_, output_);
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            if (!ggml_backend_supports_op(backend_, ggml_graph_node(graph_, i))) {
                throw std::runtime_error("Vulkan operation unsupported; fallback refused");
            }
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (!buffer_ || ggml_backend_buffer_get_size(buffer_) > 128 * 1024 * 1024) {
            throw std::runtime_error("buffer allocation failed or exceeded 128 MiB");
        }
        std::vector<float> cache(size_t(pools) * 256, 0), queries(128 * 4, 0), p(128 * 64, 0);
        std::mt19937 random(827 + pools);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        // The fixture is exactly rank 64; approximation quality is outside this timer.
        for (int key = 0; key < pools; ++key) {
            for (int d = 0; d < 64; ++d) { cache[size_t(key) * 256 + 128 + d] = normal(random); }
        }
        for (int h = 0; h < 4; ++h) {
            for (int d = 0; d < 64; ++d) { queries[h * 128 + d] = normal(random); }
        }
        for (int d = 0; d < 64; ++d) { p[d * 128 + d] = 1; }
        std::vector<int32_t> indices(pools);
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<ggml_fp16_t> masks(pools, ggml_fp32_to_fp16(0));
        const float w[4] = {1.0f/std::sqrt(128.0f), 1.0f/std::sqrt(128.0f), 1.0f/std::sqrt(128.0f), 1.0f/std::sqrt(128.0f)};
        ggml_backend_tensor_set(raw, cache.data(), 0, cache.size() * sizeof(float));
        ggml_backend_tensor_set(query, queries.data(), 0, queries.size() * sizeof(float));
        ggml_backend_tensor_set(ids, indices.data(), 0, indices.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask, masks.data(), 0, masks.size() * sizeof(ggml_fp16_t));
        if (projected) { ggml_backend_tensor_set(projection, p.data(), 0, p.size() * sizeof(float)); }
        else { ggml_backend_tensor_set(weights, w, 0, sizeof(w)); }
    }

    ~score_graph() {
        if (buffer_) { ggml_backend_buffer_free(buffer_); }
        if (ctx_) { ggml_free(ctx_); }
    }

    double run() {
        const auto start = std::chrono::steady_clock::now();
        const auto status = ggml_backend_graph_compute(backend_, graph_);
        ggml_backend_synchronize(backend_);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("score graph failed"); }
        return ms;
    }

    std::vector<int32_t> ids() {
        std::vector<int32_t> result(512);
        ggml_backend_tensor_get(output_, result.data(), 0, result.size() * sizeof(int32_t));
        std::sort(result.begin(), result.end());
        return result;
    }

private:
    ggml_backend_t backend_;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

int main() {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_name("Vulkan0");
    if (!device) { std::fprintf(stderr, "Vulkan0 absent\n"); return 1; }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { return 1; }
    int exit_code = 0;
    try {
        for (int pools : {1024, 4096, 16384, 65536}) {
            score_graph original(backend, pools, false), projected(backend, pools, true);
            for (int i = 0; i < 3; ++i) { original.run(); projected.run(); }
            std::vector<double> before, after;
            for (int i = 0; i < 15; ++i) {
                if (i % 2 == 0) { before.push_back(original.run()); after.push_back(projected.run()); }
                else { after.push_back(projected.run()); before.push_back(original.run()); }
            }
            const auto a = original.ids(), b = projected.ids();
            std::vector<int32_t> shared;
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(shared));
            std::sort(before.begin(), before.end()); std::sort(after.begin(), after.end());
            std::printf("QSA_PCA_BENCH pools=%d represented_tokens=%d stock_median_ms=%.9f rank64_median_ms=%.9f synthetic_top512_recall=%.9f\n",
                        pools, pools * 4, before[7], after[7], shared.size() / 512.0);
            std::printf("QSA_PCA_BENCH_SAMPLES pools=%d stock_ms=", pools);
            for (double x : before) { std::printf("%.9f,", x); }
            std::printf(" rank64_ms="); for (double x : after) { std::printf("%.9f,", x); } std::puts("");
        }
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); exit_code = 1; }
    ggml_backend_free(backend);
    return exit_code;
}
