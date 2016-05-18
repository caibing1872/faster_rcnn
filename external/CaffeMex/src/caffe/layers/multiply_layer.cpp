#include <vector>

#include "caffe/layers/multiply_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiplyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  transA_ = this->layer_param_.multiply_param().trans_a();
  transB_ = this->layer_param_.multiply_param().trans_b();
}

template <typename Dtype>
void MultiplyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->count(), 1) << "Size of matrix A is 0";
  CHECK_GE(bottom[1]->count(), 1) << "Size of matrix B is 0";
  CHECK_EQ(bottom[0]->count()/bottom[0]->count(0,2), 1) << "Size of matrix A should be [N C 1 1]";
  CHECK_EQ(bottom[1]->count()/bottom[0]->count(0,2), 1) << "Size of matrix B should be [N C 1 1]";
  if (transA_){
    M_ = bottom[0]->shape(1);
    K_ = bottom[0]->shape(0);
  }else{
    K_ = bottom[0]->shape(1);
    M_ = bottom[0]->shape(0);
  }
  if (transB_){
    CHECK_EQ(bottom[1]->shape(1), K_) << "Dimension of matrix A and B mismatch";
    N_ = bottom[1]->shape(0);
  }else{
    CHECK_EQ(bottom[1]->shape(0), K_) << "Dimension of matrix A and B mismatch";
    N_ = bottom[1]->shape(1);
  }
  vector<int> top_shape = bottom[0]->shape();
  top_shape[0] = M_;
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiplyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* matA = bottom[0]->cpu_data();
  const Dtype* matB = bottom[1]->cpu_data();
  Dtype* product = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(transA_?CblasTrans:CblasNoTrans, transB_?CblasTrans:CblasNoTrans, M_, N_, K_, (Dtype)1.,
      matA, matB, (Dtype)0., product);
}

template <typename Dtype>
void MultiplyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    if (transA_){
      caffe_cpu_gemm<Dtype>(CblasNoTrans, transB_?CblasNoTrans:CblasTrans, K_, M_, N_, (Dtype)1.,
        bottom[1]->cpu_data(), top_diff, (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
    }else{
      caffe_cpu_gemm<Dtype>(transB_?CblasTrans:CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
        top_diff, bottom[1]->cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
    }
  }
  if (propagate_down[1]) {
    if (transB_){
      caffe_cpu_gemm<Dtype>(CblasTrans, transA_?CblasTrans:CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom[0]->cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }else{
      caffe_cpu_gemm<Dtype>(transA_?CblasNoTrans:CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
        top_diff, bottom[0]->cpu_data(), (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiplyLayer);
#endif

INSTANTIATE_CLASS(MultiplyLayer);
REGISTER_LAYER_CLASS(Multiply);

}  // namespace caffe
