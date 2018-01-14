// ------------------------------------------------------------------
// R-FCN
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/region_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
  template <typename Dtype>
  void RegionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    RegionPoolingParameter region_pooling_param =
      this->layer_param_.region_pooling_param();
    spatial_scale_ = region_pooling_param.spatial_scale();
    LOG(INFO) << "Spatial scale: " << spatial_scale_;
    //extract features for give sampled point 
    // top 0 features shape (m, num_regions * num_features)
    // bottom 0 data (1, num_rfcn_region * num_features, H, W )
    // bottom 1 sampled points (m, 2) spatial index in roi scale
    // bottom 2 feature extracting point (num_samples, num_regions, 2) (only offset)
    // bottom 3 rfcn regions weights shape :(m, num_regions, num_samples, num_rfcn_regions) val in range(0, num_rfcn_regions)
    num_features_ = region_pooling_param.num_features();
    num_regions_ = region_pooling_param.num_regions();
    num_rfcn_regions_ = region_pooling_param.num_rfcn_regions(); 
    num_samples_ = region_pooling_param.num_samples(); 

  }

  template <typename Dtype>
  void RegionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    CHECK_EQ(channels_, num_features_ * num_rfcn_regions_)
      << "input channel number does not match layer parameters";
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(bottom[1]->num(), num_regions_, num_samples_ , num_features_);
    mapping_channel_.Reshape(bottom[1]->num(), num_regions_, num_samples_ , num_features_);
  }

  template <typename Dtype>
  void RegionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void RegionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
#ifdef CPU_ONLY
  STUB_GPU(RegionPoolingLayer);
#endif

  INSTANTIATE_CLASS(RegionPoolingLayer);
  REGISTER_LAYER_CLASS(RegionPooling);

}  // namespace caffe
