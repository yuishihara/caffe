#include <vector>

#include "caffe/layers/clipping_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClippingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ClippingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  if (Dtype(1) <= dot) {
    dot = Dtype(1);
  }
  top[0]->mutable_cpu_data()[0] = dot;
}

template <typename Dtype>
void ClippingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0];

      Dtype* clipped = diff_.mutable_cpu_data();
      const int count = bottom[i]->count();
      for (int idx = 0; idx < count; ++idx) {
        if (Dtype(1) < clipped[idx]) {
          clipped[idx] = Dtype(1);
        } else if (clipped[idx] < Dtype(-1)) {
          clipped[idx] = Dtype(-1);
        }
      }

      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ClippingLossLayer);
#endif

INSTANTIATE_CLASS(ClippingLossLayer);
REGISTER_LAYER_CLASS(ClippingLoss);

}  // namespace caffe
