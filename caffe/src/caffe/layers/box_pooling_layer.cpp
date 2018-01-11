// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/box_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void BOXPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    BOXPoolingParameter box_pooling_param =
      this->layer_param_.box_pooling_param();
    spatial_scale_ = box_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;
    //n_samples indicate the number of sampled point on each region
    CHECK_GT(box_pooling_param.output_dim(), 0)
      << "output_dim must be > 0";
    CHECK_GT(box_pooling_param.num_regions(), 0)
      << "group_size must be > 0";
    sample_idx_.Reshape(num_samples_ * num_regions_,2);
    num_features_ = box_pooling_param.num_features();
    num_regions_ = box_pooling_param.num_regions();
    num_samples_ = box_pooling_param.num_samples();
    num_rfcn_regions_ = box_pooling_param.num_rfcn_regions(); 
    FILE* f= fopen('boundaries.txt','r');
    Dtype* sample_id = sample_idx_.mutable_gpu_data();
    for (int i = 0; i < num_samples_ * num_regions_; i++ ){
      fscanf(f, "%d,%d", &sample_id[i],  &sample_id[ num_samples_ * num_regions_ + i]);
    }
    fclose(f);

  }

  template <typename Dtype>
  void BOXPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, num_features_*num_regions_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), channels_, height_, width_);
    mapping_channel_.Reshape(bottom[1]->num(), channels_, height_, width_);
  }

  template <typename Dtype>
  void BOXPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void BOXPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(BOXPoolingLayer);
#endif

  INSTANTIATE_CLASS(BOXPoolingLayer);
  REGISTER_LAYER_CLASS(BOXPooling);

}  // namespace caffe
