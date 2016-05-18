#ifndef CAFFE_SEQ_PAIR_POOL_LAYER_HPP_
#define CAFFE_SEQ_PAIR_POOL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class SeqPairPoolLayer : public Layer<Dtype> {
 public:
  explicit SeqPairPoolLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SeqPairPool"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  std::vector<int> idxa_, idxb_;
  int pair_num_;
  int cluster_num_;
  int out_feat_dim_;
  int loc_feat_dim_;
  Blob<Dtype> scaleA_, scaleB_, dataA_, dataB_, assignA_, assignB_, encodeA_, encodeB_;
  int cntA, cntB;
};

}  // namespace caffe

#endif  // CAFFE_SEQ_PAIR_POOL_LAYER_HPP_
