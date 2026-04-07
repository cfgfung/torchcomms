#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include "comms/torchcomms/device/xpu/XpuApi.hpp"

namespace torch::comms {
    int MulKernel(const xpuStream_t &stream, at::Tensor &src, const double &factor){
        try {
            sycl::queue q = stream.queue();
            auto* src_data = src.data_ptr<float>();
            int64_t numel = src.numel();

            // Explicitly set vectorization
            constexpr int vec_size = 4;
            size_t num_vectors = numel / vec_size;
            size_t remainder = numel % vec_size;

            // Normal loop 
            if (num_vectors > 0) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(num_vectors), [=](sycl::id<1> i) {
                        using vec_t = sycl::vec<float, vec_size>; 
                        auto vec_ptr = reinterpret_cast<vec_t*>(src_data);
                        vec_ptr[i] *= factor; 
                    });
                });
            }

            // Handle the remaining tail
            if (remainder > 0) {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(remainder), [=](sycl::id<1> i) {
                        size_t offset = num_vectors * vec_size;
                        src_data[offset + i[0]] *= factor; 
                    });
                });
            }
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL exception caught: " << e.what() << "\n";
            return 1;
        }

        return 0;
    }
} // namespace torch::comms
