#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void RMSPropGravesSolver<Dtype>::RMSPropGravesPreSolve() {
  // Add the extra history entries for RMSPropGraves after those from
  // SGDSolver::PreSolve
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const int ADDITIONAL_HISTORY = 2;
  for (int i = 0; i < ADDITIONAL_HISTORY; ++i) {
    for (int j = 0; j < net_params.size(); ++j) {
      const vector<int>& shape = net_params[j]->shape();
      this->history_.push_back(
              shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    }
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void rmspropgraves_update_gpu(int N, Dtype* grad, Dtype* n, Dtype* g, Dtype* delta,
    Dtype rms_decay, Dtype momentum, Dtype epsilon, Dtype local_rate);
#endif

template <typename Dtype>
void RMSPropGravesSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();

  // get the learning rate
  Dtype rms_decay = this->param_.rms_decay();
  Dtype momentum = this->param_.momentum();
  Dtype epsilon = this->param_.delta();
  Dtype local_rate = rate * net_params_lr[param_id];

  // Aliases
  size_t offset = net_params.size();
  Blob<Dtype>* val_n = this->history_[param_id].get();
  Blob<Dtype>* val_g = this->history_[param_id + offset].get();
  Blob<Dtype>* val_delta = this->history_[param_id + offset * 2].get();

  const int N = net_params[param_id]->count();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    // n <- rms_decay * n + (1 - rms_decay) * grad * grad
    // compute square of gradient in update
    caffe_powx(N,
        net_params[param_id]->cpu_diff(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    caffe_cpu_axpby(N,
        Dtype(1 - rms_decay), this->update_[param_id]->cpu_data(),
        rms_decay, val_n->mutable_cpu_data());

    // g <- rms_decay * g + (1 - rms_decay) * grad
    caffe_cpu_axpby(N,
        Dtype(1 - rms_decay), net_params[param_id]->cpu_diff(),
        rms_decay, val_g->mutable_cpu_data());

    // delta <- momentum * delta - local_rate * grad / sqrt(n - g * g + epsilon)
    // g * g
    caffe_powx(N,
        val_g->cpu_data(), Dtype(2),
        this->update_[param_id]->mutable_cpu_data());

    // n - g * g
    caffe_cpu_axpby(N,
        Dtype(1), val_n->cpu_data(),
        Dtype(-1), this->update_[param_id]-> mutable_cpu_data());

    // n - g * g + epsilon
    caffe_add_scalar(N,
        epsilon, this->update_[param_id]->mutable_cpu_data());

    // sqrt(n - g * g + epsilon)
    caffe_powx(N,
        this->update_[param_id]->cpu_data(), Dtype(0.5),
        this->update_[param_id]->mutable_cpu_data());

    // grad / sqrt(n - g * g + epsilon)
    caffe_div(N,
        net_params[param_id]->cpu_diff(),
        this->update_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    caffe_cpu_axpby(N,
        -local_rate, this->update_[param_id]->cpu_data(),
        momentum, val_delta->mutable_cpu_data());

    // copy
    caffe_cpu_scale(N,
        Dtype(-1),
        val_delta->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    rmspropgraves_update_gpu(N,
        net_params[param_id]->mutable_gpu_diff(),
        val_n->mutable_gpu_data(),
        val_g->mutable_gpu_data(),
        val_delta->mutable_gpu_data(),
        rms_decay, momentum, epsilon, local_rate);
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(RMSPropGravesSolver);
REGISTER_SOLVER_CLASS(RMSPropGraves);

}  // namespace caffe
