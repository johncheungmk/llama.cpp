#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

// Experimental constant projection weights. The original pooled-key cache and
// its native lifecycle remain authoritative. Pooled slots hold a padded projected
// prefix in this mode; external serialized state is deliberately disabled.
class qwen4exp_pca_maps {
public:
    explicit qwen4exp_pca_maps(const std::string & path);
    ggml_tensor * get(int layer) const noexcept;
    ggml_tensor * get_cpu(int layer) const noexcept;
    void place(int layer, ggml_backend_dev_t device);
    uint32_t rank() const noexcept { return rank_; }

private:
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_;
    std::vector<std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>> device_buffers_;
    std::array<ggml_tensor *, 48> maps_{};
    std::array<ggml_tensor *, 48> cpu_maps_{};
    std::array<ggml_backend_dev_t, 48> placed_devices_{};
    uint32_t rank_ = 0;
};

ggml_tensor * qwen4exp_pca_score(ggml_context * ctx, ggml_tensor * query,
        ggml_tensor * pooled, ggml_tensor * mask, float scale);
