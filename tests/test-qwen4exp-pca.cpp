#include "../src/models/qwen4exp-pca.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

static void put_u32(std::vector<unsigned char> & bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) { bytes.push_back((value >> (8 * i)) & 255); }
}

static std::vector<unsigned char> valid_map() {
    std::vector<unsigned char> bytes{'Q','P','C','A','M','A','P','1'};
    put_u32(bytes, 128); put_u32(bytes, 16); put_u32(bytes, 12);
    for (uint32_t layer = 3; layer < 48; layer += 4) {
        put_u32(bytes, layer);
        for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 128; ++col) {
                const float value = row == col ? 1.0f : 0.0f;
                uint32_t bits; std::memcpy(&bits, &value, 4); put_u32(bytes, bits);
            }
        }
    }
    return bytes;
}

static void check_scores(bool vulkan, const std::string & map_path) {
    auto * device = vulkan ? ggml_backend_dev_by_name("Vulkan0") : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!device) { throw std::runtime_error("requested test backend is absent"); }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { throw std::runtime_error("test backend init failed"); }
    try {
        qwen4exp_pca_maps maps(map_path);
        maps.place(3, device);
        maps.place(3, device); // idempotent placement must not consume more metadata
        float first[128];
        ggml_backend_tensor_get(maps.get(3), first, 0, sizeof(first));
        if (first[0] != 1.0f || first[1] != 0.0f) { throw std::runtime_error("device map placement changed values"); }
        ggml_backend_tensor_get(maps.get_cpu(3), first, 0, sizeof(first));
        if (first[0] != 1.0f || first[1] != 0.0f) { throw std::runtime_error("CPU map copy was not preserved"); }
        for (int rank : {16, 32, 64, 128}) {
            auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
            if (!ctx) { throw std::runtime_error("test context allocation failed"); }
            auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, 4, 3);
            auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, 1, 97);
            auto * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 97, 3);
            auto * score = qwen4exp_pca_score(ctx, q, k, m, 1.0f/std::sqrt(128.0f));
            auto * graph = ggml_new_graph_custom(ctx, 128, false);
            ggml_build_forward_expand(graph, score);
            for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
                if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, node))) {
                    ggml_free(ctx);
                    throw std::runtime_error("requested backend does not support a projected score op; fallback refused");
                }
            }
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) { ggml_free(ctx); throw std::runtime_error("test buffer allocation failed"); }
            std::vector<float> queries(rank * 4 * 3), keys(rank * 97), actual(97 * 3);
            std::vector<ggml_fp16_t> mask(97 * 3);
            for (size_t i = 0; i < queries.size(); ++i) { queries[i] = std::sin(float(i) * 0.031f); }
            for (size_t i = 0; i < keys.size(); ++i) { keys[i] = std::cos(float(i) * 0.017f); }
            for (int token = 0; token < 3; ++token) {
                const int visible = token == 0 ? 29 : token == 1 ? 59 : 97;
                for (int key = 0; key < 97; ++key) { mask[token * 97 + key] = ggml_fp32_to_fp16(key < visible ? 0.0f : -INFINITY); }
            }
            ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
            ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(float));
            ggml_backend_tensor_set(m, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
            const auto status = ggml_backend_graph_compute(backend, graph);
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(score, actual.data(), 0, actual.size() * sizeof(float));
            double max_error = 0;
            bool valid = status == GGML_STATUS_SUCCESS;
            for (int token = 0; token < 3; ++token) {
                const int visible = token == 0 ? 29 : token == 1 ? 59 : 97;
                for (int key = 0; key < 97; ++key) {
                    if (key >= visible) {
                        valid = valid && actual[token * 97 + key] == -INFINITY;
                        continue;
                    }
                    double expected = 0;
                    for (int head = 0; head < 4; ++head) {
                        double dot = 0;
                        for (int d = 0; d < rank; ++d) { dot += double(queries[(token * 4 + head) * rank + d]) * keys[key * rank + d]; }
                        expected += std::max(0.0, dot);
                    }
                    const double error = std::abs(actual[token * 97 + key] - expected / std::sqrt(128.0));
                    valid = valid && std::isfinite(error) && error <= 1e-3;
                    max_error = std::max(max_error, error);
                }
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            if (!valid) { throw std::runtime_error("projected QSA score/mask numerical check failed"); }
            std::printf("QSA_PCA_SCORE_PASS backend=%s rank=%d max_abs_error=%.9g\n", ggml_backend_name(backend), rank, max_error);
        }
    } catch (...) {
        ggml_backend_free(backend);
        throw;
    }
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const auto path = std::filesystem::temp_directory_path() /
        ("qsa-pca-test-" + std::to_string(reinterpret_cast<uintptr_t>(&main)) + ".bin");
    auto check = [&](const std::vector<unsigned char> & bytes, bool expect_valid) {
        { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()); }
        bool accepted = false;
        try {
            qwen4exp_pca_maps maps(path.string());
            accepted = maps.rank() == 16 && maps.get(3) && maps.get(47) && !maps.get(0) && !maps.get(-1) && !maps.get(48);
            if (accepted) {
                float row[128]; ggml_backend_tensor_get(maps.get(3), row, 0, sizeof(row));
                accepted = row[0] == 1.0f && row[1] == 0.0f;
            }
        } catch (const std::exception &) {}
        if (accepted != expect_valid) { throw std::runtime_error("QSA PCA loader acceptance mismatch"); }
    };
    try {
        const auto good = valid_map();
        check(good, true);
        auto bad = good; bad[0] = '!'; check(bad, false);
        bad = good; bad[8] = 127; check(bad, false); // dimension
        bad = good; bad[12] = 255; check(bad, false); // rank
        bad = good; bad[16] = 255; check(bad, false); // layer count
        bad = good; bad[20] = 7; check(bad, false); // layer order
        bad = good; bad.resize(32); check(bad, false);
        bad = good; bad.push_back(0); check(bad, false);
        bad = good; bad[24] = 0; bad[25] = 0; bad[26] = 128; bad[27] = 127; check(bad, false); // +inf
        bad = good; bad[24] = bad[25] = bad[26] = bad[27] = 0; check(bad, false); // non-unit row
        bad.assign(1024 * 1024 + 1, 0); check(bad, false);
        { std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char *>(good.data()), good.size()); }
        check_scores(argc == 2 && std::string(argv[1]) == "--vulkan", path.string());
    } catch (const std::exception & error) {
        std::filesystem::remove(path);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::filesystem::remove(path);
    std::puts("QSA_PCA_NATIVE_LOADER_PASS");
    return 0;
}
