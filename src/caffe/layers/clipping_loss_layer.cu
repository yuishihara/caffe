#include <vector>

#include "caffe/layers/clipping_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Clip(int N, Dtype* clipped) {
  CUDA_KERNEL_LOOP(i, N) {
    if (Dtype(1) < clipped[i]) {
      clipped[i] = Dtype(1);
    } else if (clipped[i] < Dtype(-1)) {
      clipped[i] = Dtype(-1);
    }
  }
}

template <typename Dtype>
void ClippingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  if (Dtype(1) <= dot) {
    dot = Dtype(1);
  }
  Dtype loss = dot;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ClippingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];

      const int N = bottom[i]->count();
      Clip<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
                  N, diff_.mutable_gpu_data());
      CUDA_POST_KERNEL_CHECK;

      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ClippingLossLayer);

}  // namespace caffe
