#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void RMSPropGravesUpdate(int N, Dtype* grad, Dtype* n, Dtype* g, Dtype* delta,
    Dtype rms_decay, Dtype momentum, Dtype epsilon, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gradi = grad[i];
    float ni = n[i] = rms_decay * n[i] + (1 - rms_decay) * gradi * gradi;
    float gi = g[i] = rms_decay * g[i] + (1 - rms_decay) * gradi;
    float di = delta[i] = momentum * delta[i] -
        local_rate * gradi / sqrt(ni - gi * gi + epsilon);
    grad[i] = -di;
  }
}
template <typename Dtype>
void rmspropgraves_update_gpu(int N, Dtype* grad, Dtype* n, Dtype* g, Dtype* delta,
    Dtype rms_decay, Dtype momentum, Dtype epsilon, Dtype local_rate) {
  RMSPropGravesUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, grad, n, g, delta, rms_decay, momentum, epsilon, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void rmspropgraves_update_gpu<float>(int, float*, float*, float*, float*,
    float, float, float, float);
template void rmspropgraves_update_gpu<double>(int, double*, double*, double*, double*,
    double, double, double, double);

}  // namespace caffe
