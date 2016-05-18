#include <vector>

#include "caffe/layers/multiply_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiplyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* matA = bottom[0]->gpu_data();
  const Dtype* matB = bottom[1]->gpu_data();
  Dtype* product = top[0]->mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(transA_?CblasTrans:CblasNoTrans, transB_?CblasTrans:CblasNoTrans, M_, N_, K_, (Dtype)1.,
      matA, matB, (Dtype)0., product);
}

template <typename Dtype>
void MultiplyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    if (transA_){
      caffe_gpu_gemm<Dtype>(CblasNoTrans, transB_?CblasNoTrans:CblasTrans, K_, M_, N_, (Dtype)1.,
        bottom[1]->gpu_data(), top_diff, (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
    }else{
      caffe_gpu_gemm<Dtype>(transB_?CblasTrans:CblasNoTrans, CblasTrans, M_, K_, N_, (Dtype)1.,
        top_diff, bottom[1]->gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
    }
  }
  if (propagate_down[1]) {
    if (transB_){
      caffe_gpu_gemm<Dtype>(CblasTrans, transA_?CblasTrans:CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom[0]->gpu_data(), (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }else{
      caffe_gpu_gemm<Dtype>(transA_?CblasNoTrans:CblasTrans, CblasNoTrans, K_, N_, M_, (Dtype)1.,
        top_diff, bottom[0]->gpu_data(), (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiplyLayer);

}  // namespace caffe
