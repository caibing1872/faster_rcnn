#include <vector>

#include "caffe/filler.hpp"
#include "boost/shared_ptr.hpp"
#include "caffe/layers/seq_pair_pool_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/*Input:  1.feature_map: [m+n, c, h, w]
          2.label:[m+n, 1, h, w] belongs to {0, 1}
          3.assign:[m+n, 4, h, w]
*/

template <typename Dtype>
void SeqPairPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  cluster_num_ = bottom[2]->shape(1);
  scaleA_.Reshape(cluster_num_, 1, 1, 1);
  scaleB_.Reshape(cluster_num_, 1, 1, 1);
  LOG(INFO) << "Cluster number is " << cluster_num_;
  pair_num_ = cluster_num_*2;
}

template <typename Dtype>
void SeqPairPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  idxa_.clear();
  idxb_.clear();
  cntA = cntB = 0;
  // Figure out the dimensions
  loc_feat_dim_ = bottom[0]->shape(1);
  //LOG(INFO) << "loc_feat_dim is " << loc_feat_dim_;
  out_feat_dim_ = 2 * loc_feat_dim_ * cluster_num_;
  //LOG(INFO) << "out_feat_dim is " << out_feat_dim_;
  vector<int> top_shape = bottom[0]->shape();
  top_shape[0] = 2;
  top_shape[1] = out_feat_dim_;
  top[0]->Reshape(top_shape);
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void SeqPairPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[1]->shape(0), bottom[2]->shape(0));
  CHECK_EQ(bottom[2]->shape(1), cluster_num_);
  const int batch_size = bottom[0]->shape(0);
  //LOG(INFO) << "Batch size = " << batch_size;
  // bool has_dim = false;
  if (bottom[2]->count() > bottom[2]->shape(0)*bottom[2]->shape(1)){
    NOT_IMPLEMENTED;
    CHECK_EQ(bottom[2]->shape(2), cluster_num_);
    CHECK_EQ(bottom[2]->shape(3), cluster_num_);
    // has_dim = true;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* assign = bottom[2]->cpu_data();
  for (int i = 0; i < batch_size; ++i){
    //LOG(INFO) << "Label[" << i << "] = " << label[i];
    if (label[i] == 0){
      idxa_.push_back(i);
      cntA = cntA + 1;
      //LOG(INFO) << "cntA = "<< cntA;
    }else if(label[i] == 1){
      idxb_.push_back(i);
      cntB = cntB + 1;
      //LOG(INFO) << "cntB = "<< cntB;
    }else{
      LOG(FATAL) << "seq pair pool only recept 0 and 1 label.";
    }
  }
  //LOG(INFO) << "Count: cntA:" << cntA << " cntB:" << cntB;
  CHECK_GE(cntA, 1) << "At least one frame should in seq A";
  CHECK_GE(cntB, 1) << "At least one frame should in seq B";
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch(this->layer_param_.seq_pair_pool_param().pool()){
    case SeqPairPoolParameter_PoolMethod_AVE:
    {
      dataA_.Reshape(cntA, loc_feat_dim_, 1, 1);
      dataB_.Reshape(cntB, loc_feat_dim_, 1, 1);
      assignA_.Reshape(cntA, cluster_num_, 1, 1);
      assignB_.Reshape(cntB, cluster_num_, 1, 1);
      encodeA_.Reshape(cluster_num_, loc_feat_dim_, 1, 1);
      encodeB_.Reshape(cluster_num_, loc_feat_dim_, 1, 1);
      int data_bias_A, data_bias_B, assign_bias_A, assign_bias_B;
      data_bias_A=data_bias_B=assign_bias_A=assign_bias_B=0;
      for (int i = 0; i < batch_size; ++i){
        if (label[i] == 0)
        {
          //LOG(INFO) << "Copy " << i << "/" << batch_size-1 << "to A";
          caffe_copy<Dtype>(loc_feat_dim_, bottom_data + i * loc_feat_dim_, dataA_.mutable_cpu_data() + data_bias_A);
          data_bias_A += loc_feat_dim_;
          caffe_copy<Dtype>(cluster_num_, assign + i * cluster_num_, assignA_.mutable_cpu_data() + assign_bias_A);
          //LOG(INFO) << "Copy " << i << "/" << batch_size-1 << "to A Done";
          assign_bias_A += cluster_num_;
        }else{
          //LOG(INFO) << "Copy " << i << "/" << batch_size-1 << "to B";
          caffe_copy<Dtype>(loc_feat_dim_, bottom_data + i * loc_feat_dim_, dataB_.mutable_cpu_data() + data_bias_B);
          data_bias_B += loc_feat_dim_;
          caffe_copy<Dtype>(cluster_num_, assign + i * cluster_num_, assignB_.mutable_cpu_data() + assign_bias_B);
          //LOG(INFO) << "Copy " << i << "/" << batch_size-1 << "to B Done";
          assign_bias_B += cluster_num_;
        }
      }
      // multiply
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, cluster_num_, loc_feat_dim_, cntA, 1, assignA_.cpu_data(), dataA_.cpu_data(), 0, encodeA_.mutable_cpu_data());
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, cluster_num_, loc_feat_dim_, cntB, 1, assignB_.cpu_data(), dataB_.cpu_data(), 0, encodeB_.mutable_cpu_data());
      // sum
      // LOG(INFO) << "Assigna:";
      // for (int ttt = 0; ttt < cntA*cluster_num_; ttt++){
      //   LOG(INFO) << assignA_.cpu_data()[ttt];
      // }
      if (false){
        shared_ptr<Blob<Dtype> > sum_buff(new Blob<Dtype>(1, 1, 1, 1));
        sum_buff->Reshape(cntA, 1, 1, 1);
        caffe_set(sum_buff->count(), Dtype(1), sum_buff->mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, cntA, loc_feat_dim_, 1, assignA_.cpu_data(), sum_buff->cpu_data(), 0, scaleA_.mutable_cpu_data());
        sum_buff->Reshape(cntB, 1, 1, 1);
        caffe_set(sum_buff->count(), Dtype(1), sum_buff->mutable_cpu_data());
        caffe_cpu_gemv<Dtype>(CblasTrans, cntB, loc_feat_dim_, 1, assignB_.cpu_data(), sum_buff->cpu_data(), 0, scaleB_.mutable_cpu_data());
      }else{
        // LOG(INFO) << "Count: cntA:" << cntA << " cntB:" << cntB;
        for (int clust_t = 0; clust_t < cluster_num_; ++clust_t){
          scaleA_.mutable_cpu_data()[clust_t] = 0;
          scaleB_.mutable_cpu_data()[clust_t] = 0;
          for (int num_t = 0; num_t < cntA; ++num_t)
          {
            // LOG(INFO) << "scaleA: " << scaleA_.mutable_cpu_data()[clust_t] << " -> ";
            scaleA_.mutable_cpu_data()[clust_t] += assignA_.cpu_data()[num_t*cluster_num_+clust_t];
            // LOG(INFO) << scaleA_.mutable_cpu_data()[clust_t];
          }
          for (int num_t = 0; num_t < cntB; ++num_t)
          {
            scaleB_.mutable_cpu_data()[clust_t] += assignB_.cpu_data()[num_t*cluster_num_+clust_t];
          }
        }
      }
      //division
      for (int i = 0; i < cluster_num_; ++i){
         //LOG(INFO) << "ScaleA = " << scaleA_.cpu_data()[i];
         //LOG(INFO) << "ScaleB = " << scaleB_.cpu_data()[i];
        for (int iter = 0; iter < loc_feat_dim_  ; ++iter)
        {
          // LOG(INFO) << "Scale: " << *(encodeA_.mutable_cpu_data() + i * loc_feat_dim_ + iter) << " to ";
          *(encodeA_.mutable_cpu_data() + i * loc_feat_dim_ + iter) /= (scaleA_.cpu_data()[i]+1e-4);
          *(encodeB_.mutable_cpu_data() + i * loc_feat_dim_ + iter) /= (scaleB_.cpu_data()[i]+1e-4);
          // LOG(INFO) << *(encodeA_.mutable_cpu_data() + i * loc_feat_dim_ + iter);
        }
        // caffe_cpu_axpby<Dtype>(loc_feat_dim_, Dtype(1)/scaleA_.cpu_data()[i], encodeA_.mutable_cpu_data() + i * loc_feat_dim_, Dtype(0), encodeA_.mutable_cpu_data() + i * loc_feat_dim_);
        // caffe_cpu_axpby<Dtype>(loc_feat_dim_, Dtype(1)/scaleB_.cpu_data()[i], encodeB_.mutable_cpu_data() + i * loc_feat_dim_, Dtype(0), encodeB_.mutable_cpu_data() + i * loc_feat_dim_);
        caffe_copy(loc_feat_dim_, encodeA_.cpu_data()+ i * loc_feat_dim_, top_data + i * loc_feat_dim_);
        caffe_copy(loc_feat_dim_, encodeA_.cpu_data()+ i * loc_feat_dim_, top_data + 3*out_feat_dim_/2 + i * loc_feat_dim_);
        caffe_copy(loc_feat_dim_, encodeB_.cpu_data()+ i * loc_feat_dim_, top_data + out_feat_dim_/2 + i * loc_feat_dim_);
        caffe_copy(loc_feat_dim_, encodeB_.cpu_data()+ i * loc_feat_dim_, top_data + out_feat_dim_ + i * loc_feat_dim_);
      }
      
      // for (int frame = 0; frame < batch_size; ++frame){
      //   int label_t = (int)label[frame];
      //   int base_b1 = label_t * cluster_num_ * loc_feat_dim_;
      //   int base_b2 = (1-label_t) * cluster_num_ * loc_feat_dim_;
      //   for (int loc_feat_dim_ = 0; loc_feat_dim_ < loc_feat_dim_; ++loc_feat_dim_)
      //   {
      //     Dtype data_t = bottom_data[frame*loc_feat_dim_+loc_feat_dim_];
      //     for (int cluster = 0; cluster < cluster_num_; ++cluster)
      //     {
      //       top_data[base_b1+cluster*loc_feat_dim_+loc_feat_dim_] = data_t * assign[frame*cluster_num_+cluster];
      //       top_data[out_feat_dim_+base_b2+cluster*loc_feat_dim_+loc_feat_dim_] = data_t * assign[frame*cluster_num_+cluster];
      //     }
      //   }
      // }
      break;
    }
    case SeqPairPoolParameter_PoolMethod_MAX:
      NOT_IMPLEMENTED;
      break;
    case SeqPairPoolParameter_PoolMethod_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown seq pair pool type";
  }

}

template <typename Dtype>
void SeqPairPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int bias_t = out_feat_dim_/2;
    for (int i = 0; i < bias_t; ++i)
    {
      top[0]->mutable_cpu_diff()[i] += top[0]->cpu_diff()[3*bias_t + i];
      top[0]->mutable_cpu_diff()[bias_t+i] += top[0]->cpu_diff()[2*bias_t + i];
    }

    // 1. Calculate data's diff
    Blob<Dtype> temp_buff;
    temp_buff.ReshapeLike(*(top[0]));
    temp_buff.CopyFrom(*(top[0]), true);
    for (int i = 0; i < cluster_num_; ++i){
      for (int iter = 0; iter < loc_feat_dim_  ; ++iter)
      {
        *(temp_buff.mutable_cpu_diff() + i * loc_feat_dim_ + iter) /= scaleA_.cpu_data()[i];
        *(temp_buff.mutable_cpu_diff() + i * loc_feat_dim_ + iter + out_feat_dim_ / 2) /= scaleB_.cpu_data()[i];
      }
    }
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, cntA, loc_feat_dim_, cluster_num_, (Dtype)1.,
        temp_buff.cpu_diff(), dataA_.cpu_data(), (Dtype)0.,
        dataA_.mutable_cpu_diff());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, cntB, loc_feat_dim_, cluster_num_, (Dtype)1.,
        temp_buff.cpu_diff() + out_feat_dim_/2, dataB_.cpu_data(), (Dtype)0.,
        dataB_.mutable_cpu_diff());
    // 2. Calculate assign's diff
    for (int sample = 0; sample < cntA; ++sample)
    {
      for (int clust_t = 0; clust_t < cluster_num_; ++clust_t)
      {
        Blob<Dtype> temp_buff;
        std::vector<int> temp_shape(1, loc_feat_dim_);
        temp_buff.Reshape(temp_shape);
        caffe_sub<Dtype>(loc_feat_dim_, dataA_.cpu_data()+sample*loc_feat_dim_, top[0]->cpu_data()+clust_t*loc_feat_dim_, temp_buff.mutable_cpu_data());
        assignA_.mutable_cpu_diff()[sample*cluster_num_+clust_t] = caffe_cpu_dot(loc_feat_dim_, top[0]->cpu_diff()+clust_t*loc_feat_dim_, temp_buff.cpu_data()) / scaleA_.cpu_data()[clust_t];
      }
    }
    for (int sample = 0; sample < cntB; ++sample)
    {
      for (int clust_t = 0; clust_t < cluster_num_; ++clust_t)
      {
        Blob<Dtype> temp_buff;
        std::vector<int> temp_shape(1, loc_feat_dim_);
        temp_buff.Reshape(temp_shape);
        caffe_sub<Dtype>(loc_feat_dim_, dataB_.cpu_data()+sample*loc_feat_dim_, top[0]->cpu_data()+out_feat_dim_/2+clust_t*loc_feat_dim_, temp_buff.mutable_cpu_data());
        assignB_.mutable_cpu_diff()[sample*cluster_num_+clust_t] = caffe_cpu_dot(loc_feat_dim_, top[0]->cpu_diff()+out_feat_dim_/2+clust_t*loc_feat_dim_, temp_buff.cpu_data()) / scaleB_.cpu_data()[clust_t];
      }
    }
    // 3. Distribute diff to bottom
    for (int i = 0; i < cntA; ++i)
    {
      caffe_copy(loc_feat_dim_, dataA_.cpu_diff() + i * loc_feat_dim_, bottom[0]->mutable_cpu_diff() + idxa_[i] * loc_feat_dim_);
      caffe_copy(cluster_num_, assignA_.cpu_diff() + i * cluster_num_, bottom[2]->mutable_cpu_diff() + idxa_[i] * cluster_num_);
    }
    for (int i = 0; i < cntB; ++i)
    {
      caffe_copy(loc_feat_dim_, dataB_.cpu_diff() + i * loc_feat_dim_, bottom[0]->mutable_cpu_diff() + idxb_[i] * loc_feat_dim_);
      caffe_copy(cluster_num_, assignA_.cpu_diff() + i * cluster_num_, bottom[2]->mutable_cpu_diff() + idxb_[i] * cluster_num_);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SeqPairPoolLayer);
#endif

INSTANTIATE_CLASS(SeqPairPoolLayer);
REGISTER_LAYER_CLASS(SeqPairPool);

}  // namespace caffe
