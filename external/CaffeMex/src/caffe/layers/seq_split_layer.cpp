#include <vector>

#include "caffe/layers/seq_split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SeqSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  label_a_ = 0;
  label_b_ = 1;
  cnt_a_ = 0;
  cnt_b_ = 0;
  total_num_ = 0;
  a_atbottom_.clear();
  b_atbottom_.clear();
}

template <typename Dtype>
void SeqSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  k_ = this->layer_param_.seq_split_param().k();
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  const Dtype* label = bottom[1]->cpu_data();
  for(int i = 1; i < bottom[0]->shape(0); i++){
    if(label[i] == label_a_){
      cnt_a_++;
    }else if(label[i] == label_b_){
      cnt_b_++;
    }else{
      LOG(FATAL) << "Label into SeqSplitLayer can only be 0 or 1";
    }
  }
  vector<int> top_shape_a = bottom[0]->shape();
  vector<int> top_shape_b = bottom[0]->shape();
  switch (this->layer_param_.seq_split_param().select()){
    case SeqSplitParameter_SelectWay_ALL:
      top_shape_a[0] = cnt_a_;
      top_shape_b[0] = cnt_b_;
      total_num_ = cnt_a_ + cnt_b_;
      k_ = -1;
      break;
    case SeqSplitParameter_SelectWay_RANDOM_K:
      CHECK_GE(k_, 0);
      top_shape_a[0] = k_;
      top_shape_b[0] = k_;
      total_num_ = 2 * k_;
      break;
    case SeqSplitParameter_SelectWay_RANDOM_MIN:
      k_ = cnt_a_ < cnt_b_ ? cnt_a_ : cnt_b_;
      top_shape_a[0] = k_;
      top_shape_b[0] = k_;
      total_num_ = 2 * k_;
      break;
    default:
      LOG(FATAL) << "Unknown select type";
  }
  top[0]->Reshape(top_shape_a);
  top[1]->Reshape(top_shape_b);
  a_atbottom_.resize(cnt_a_);
  b_atbottom_.resize(cnt_b_);
}

template <typename Dtype>
void SeqSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->shape(0), 2*k_);
  int N = bottom[1]->shape(0);
  const Dtype* data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int top_offset_a = 0;
  int top_offset_b = 0;
  Dtype* top_a = top[0]->mutable_cpu_data();
  Dtype* top_b = top[1]->mutable_cpu_data();
  const int slice_num = bottom[0]->count(1);
  int cnt_a = 0;
  int cnt_b = 0;
  switch (this->layer_param_.seq_split_param().select()){
    case SeqSplitParameter_SelectWay_ALL:
      for (int i = 0; i < N; i++){
        if (label[i] == label_a_){
          caffe_copy(slice_num, data + i*slice_num, top_a + top_offset_a);
          top_offset_a += slice_num;
          a_atbottom_[cnt_a] = i;
          cnt_a++;
        }else{
          caffe_copy(slice_num, data + i*slice_num, top_b + top_offset_b);
          top_offset_b += slice_num;
          a_atbottom_[cnt_b] = i;
          cnt_b++;
        }
      }
      break;
    case SeqSplitParameter_SelectWay_RANDOM_K:
      CHECK_GE(cnt_a_, k_);
      CHECK_GE(cnt_b_, k_);
      for (int i = 0; i < N; i++){
        if (label[i] == label_a_ && cnt_a < k_){
          caffe_copy(slice_num, data + i*slice_num, top_a + top_offset_a);
          top_offset_a += slice_num;
          a_atbottom_[cnt_a] = i;
          cnt_a++;
        }else if(cnt_b < k_){
          caffe_copy(slice_num, data + i*slice_num, top_b + top_offset_b);
          top_offset_b += slice_num;
          a_atbottom_[cnt_b] = i;
          cnt_b++;
        }
      }
      break;
    case SeqSplitParameter_SelectWay_RANDOM_MIN:
      for (int i = 0; i < N; i++){
        if (label[i] == label_a_ && cnt_a < k_){
          caffe_copy(slice_num, data + i*slice_num, top_a + top_offset_a);
          top_offset_a += slice_num;
          a_atbottom_[cnt_a] = i;
          cnt_a++;
        }else if(cnt_b < k_){
          caffe_copy(slice_num, data + i*slice_num, top_b + top_offset_b);
          top_offset_b += slice_num;
          a_atbottom_[cnt_b] = i;
          cnt_b++;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown select type";
  }
}

template <typename Dtype>
void SeqSplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int slice_num = bottom[0]->count(1);
    const Dtype* top_diff_a = top[0]->cpu_diff();
    const Dtype* top_diff_b = top[1]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int i = 0; i < cnt_a_; i++){
      caffe_copy(slice_num, top_diff_a + i * slice_num, bottom_diff + a_atbottom_[i] * slice_num);
    }
    for (int i = 0; i < cnt_b_; ++i){
      caffe_copy(slice_num, top_diff_b + i * slice_num, bottom_diff + b_atbottom_[i] * slice_num);
    }
  }
}



INSTANTIATE_CLASS(SeqSplitLayer);
REGISTER_LAYER_CLASS(SeqSplit);

}  // namespace caffe
