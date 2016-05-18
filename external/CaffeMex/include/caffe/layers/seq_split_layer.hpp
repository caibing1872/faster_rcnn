#ifndef CAFFE_SEQSPLIT_LAYER_HPP_
#define CAFFE_SEQSPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 *
 * Note: similarly to FlattenLayer, this layer does not change the input values
 * (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
 */
template <typename Dtype>
class SeqSplitLayer : public Layer<Dtype> {
 public:
  explicit SeqSplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SeqSplit"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  /// @brief vector of axes indices whose dimensions we'll copy from the bottom
  int k_, total_num_;
  Dtype label_a_, label_b_;
  int cnt_a_, cnt_b_;
  std::vector<int> a_atbottom_, b_atbottom_;
};

}  // namespace caffe

#endif  // CAFFE_SEQSPLIT_LAYER_HPP_
