
#pragma once
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include "comms/torchcomms/device/xpu/XpuApi.hpp"

namespace torch::comms {
    // Create a new tensor -- at::empty_like, uninitialized
    // Read the memory and multiply it into the new tensor
    int CloneAndMulKernel(const xpuStream_t &stream, const at::Tensor &src, at::Tensor &dst, const double &factor);
} // namespace torch::comms
