#include "qwen4exp-pca.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static uint32_t qwen_pca_u32(std::istream & input) {
    unsigned char bytes[4];
    if (!input.read(reinterpret_cast<char *>(bytes), 4)) {
        throw std::runtime_error("QSA PCA: truncated map");
    }
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
           (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

qwen4exp_pca_maps::qwen4exp_pca_maps(const std::string & path) :
    ctx_(nullptr, ggml_free), buffer_(nullptr, ggml_backend_buffer_free) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 20 || input.tellg() > 1024 * 1024) {
        throw std::runtime_error("QSA PCA: map must be a readable file of 20 bytes to 1 MiB");
    }
    input.seekg(0);
    char magic[8];
    if (!input.read(magic, 8) || std::memcmp(magic, "QPCAMAP1", 8) != 0) {
        throw std::runtime_error("QSA PCA: invalid map magic/version");
    }
    const uint32_t dimension = qwen_pca_u32(input);
    rank_ = qwen_pca_u32(input);
    const uint32_t layers = qwen_pca_u32(input);
    if (dimension != 128 || layers != 12 || (rank_ != 16 && rank_ != 32 && rank_ != 64 && rank_ != 128)) {
        throw std::runtime_error("QSA PCA: expected dimension 128, 12 layers, rank 16/32/64/128");
    }
    const size_t count = size_t(rank_) * dimension;
    std::array<std::vector<float>, 12> values;
    for (uint32_t index = 0; index < layers; ++index) {
        if (qwen_pca_u32(input) != 3 + 4 * index) {
            throw std::runtime_error("QSA PCA: layers must be ordered 3,7,...,47");
        }
        auto & projection = values[index];
        projection.resize(count);
        for (float & value : projection) {
            const uint32_t bits = qwen_pca_u32(input);
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value)) {
                throw std::runtime_error("QSA PCA: non-finite projection");
            }
        }
        for (uint32_t row = 0; row < rank_; ++row) {
            for (uint32_t other = 0; other <= row; ++other) {
                double dot = 0;
                for (uint32_t col = 0; col < dimension; ++col) {
                    dot += double(projection[row * dimension + col]) * projection[other * dimension + col];
                }
                if (std::abs(dot - (row == other ? 1.0 : 0.0)) > 1e-3) {
                    throw std::runtime_error("QSA PCA: projection rows are not orthonormal");
                }
            }
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("QSA PCA: unexpected trailing data");
    }
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("QSA PCA: CPU backend is required for constant map storage");
    }
    ctx_.reset(ggml_init({24 * ggml_tensor_overhead() + 4096, nullptr, true}));
    if (!ctx_) {
        throw std::runtime_error("QSA PCA: metadata allocation failed");
    }
    for (uint32_t index = 0; index < layers; ++index) {
        const uint32_t layer = 3 + 4 * index;
        maps_[layer] = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, dimension, rank_);
        cpu_maps_[layer] = maps_[layer];
        ggml_format_name(maps_[layer], "experimental_qsa_pca-%u", layer);
    }
    buffer_.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), ggml_backend_dev_buffer_type(cpu)));
    if (!buffer_) {
        throw std::runtime_error("QSA PCA: constant buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (uint32_t index = 0; index < layers; ++index) {
        ggml_backend_tensor_set(maps_[3 + 4 * index], values[index].data(), 0, count * sizeof(float));
    }
}

ggml_tensor * qwen4exp_pca_maps::get(int layer) const noexcept {
    return layer >= 0 && layer < int(maps_.size()) ? maps_[layer] : nullptr;
}

ggml_tensor * qwen4exp_pca_maps::get_cpu(int layer) const noexcept {
    return layer >= 0 && layer < int(cpu_maps_.size()) ? cpu_maps_[layer] : nullptr;
}

void qwen4exp_pca_maps::place(int layer, ggml_backend_dev_t device) {
    ggml_tensor * original = get(layer);
    if (!original || !device) {
        throw std::runtime_error("QSA PCA: invalid projection placement");
    }
    if (placed_devices_[layer]) {
        if (placed_devices_[layer] != device) {
            throw std::runtime_error("QSA PCA: projection device placement is immutable");
        }
        return;
    }
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        placed_devices_[layer] = device;
        return;
    }
    std::vector<float> values(size_t(128) * rank_);
    ggml_backend_tensor_get(original, values.data(), 0, values.size() * sizeof(float));
    auto * replacement = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, 128, rank_);
    ggml_format_name(replacement, "experimental_qsa_pca-%d", layer);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
            ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), ggml_backend_dev_buffer_type(device)),
            ggml_backend_buffer_free);
    if (!storage) {
        throw std::runtime_error("QSA PCA: device projection allocation failed");
    }
    ggml_backend_buffer_set_usage(storage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(replacement, values.data(), 0, values.size() * sizeof(float));
    device_buffers_.push_back(std::move(storage));
    maps_[layer] = replacement;
    placed_devices_[layer] = device;
}

ggml_tensor * qwen4exp_pca_score(ggml_context * ctx, ggml_tensor * query,
        ggml_tensor * pooled, ggml_tensor * mask, float scale) {
    ggml_tensor * keys = ggml_reshape_2d(ctx, pooled, pooled->ne[0], pooled->ne[2]);
    ggml_tensor * sum = nullptr;
    for (int64_t head = 0; head < query->ne[1]; ++head) {
        ggml_tensor * q = ggml_view_2d(ctx, query, query->ne[0], query->ne[2], query->nb[2], head * query->nb[1]);
        ggml_tensor * score = ggml_mul_mat(ctx, keys, q);
        ggml_prec_set_acc(score, GGML_PREC_F32);
        ggml_prec_set_src(score, GGML_PREC_F16, 1);
        score = ggml_relu(ctx, score);
        sum = sum ? ggml_add(ctx, sum, score) : score;
    }
    return ggml_add(ctx, ggml_scale(ctx, sum, scale), ggml_cast(ctx, mask, GGML_TYPE_F32));
}
