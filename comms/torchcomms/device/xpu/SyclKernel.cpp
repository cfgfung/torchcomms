#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include "comms/torchcomms/device/xpu/XpuApi.hpp"

namespace torch::comms {
    int CloneAndMulKernel(const xpuStream_t &stream, const at::Tensor &src, at::Tensor &dst, const double &factor){
        try {
            sycl::queue q = stream.queue();
            auto* src_data = src.data_ptr<float>();
            auto* dst_data = dst.data_ptr<float>();
            int64_t numel = src.numel();
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                    dst_data[i] = src_data[i] * factor; 
                });
            });
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught: " << e.what() << "\n";
            return 1;
        }

        return 0;
    }
} // namespace torch::comms
