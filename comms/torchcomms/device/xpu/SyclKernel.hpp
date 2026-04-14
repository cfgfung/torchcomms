
#pragma once
#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include "comms/torchcomms/device/xpu/XpuApi.hpp"

namespace torch::comms {
    // Read the memory and perform inplace multiplication
    int MulKernel(const xpuStream_t &stream, at::Tensor &src, const double &factor);
    int DivKernel(const xpuStream_t &stream, at::Tensor &src, const float &factor, const bool& trunc);
} // namespace torch::comms
