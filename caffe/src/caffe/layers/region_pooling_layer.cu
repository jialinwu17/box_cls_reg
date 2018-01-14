#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>
#include "caffe/layers/region_pooling_layer.hpp"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void BOXPoolingForward(
    const int nthreads,
    const int num_features, const int num_regions,const int num_samples, const int M, const Dtype * feat_ext_offset,
    const Dtype* bottom_data,
    const int height, const int width,
    Dtype* top_data,
    const int num_rfcn_regions,
    int* mapping_channel,
    const Dtype * seed_points,
    const Dtype* rfcn_regions_weights) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The input is in order (1, num_rfcn_regions * num_features , H, W ), nf = 64
      // seed_points is in order (m, 2)
      // feat_ext_offset is in order (m, num_regions, num_samples, 2)
      // rfcn_region_idx
      // output is in order (m, num_regions, num_samples, num_rfcn_regions, num_features)
      // nthreads m * num_regions * num_features

      for (int rfcn_r = 0 ; rfcn_r < num_rfcn_regions ; rfcn_r ++){
      int output_dim = num_features * num_regions;
      int f = index % num_features;
      int s = (index / num_features ) % num_samples;
      int r = (index / num_features / num_samples) % num_regions;
      int m = index / num_features  / num_samples / num_regions;
      int x = seed_points[ 2 * m ];
      int y = seed_points[ 2 * m + 1 ];
      int feat_ext_idx = feat_ext_offset [ m * num_regions * num_samples *2 +  2 * r * num_samples + 2 * s ] + x;
      int feat_ext_idy = feat_ext_offset [ m * num_regions * num_samples *2 +  2 * r * num_samples + 2 * s + 1] + y;
      int bottom_idx = ((rfcn_r * num_features + f) * height + feat_ext_idy) * width + feat_ext_idx;
      int map_channel_idx = ((f) * height + feat_ext_idy) * width + feat_ext_idx;
      int rfcn_regions_weights_idx = (((m * num_regions + r)* num_samples + s) * num_rfcn_regions + rfcn_r) ;
      top_data[index] += bottom_data[bottom_idx] * rfcn_regions_weights[rfcn_regions_weights_idx];
      mapping_channel[ index ] = map_channel_idx;
    }
    top_data[index] /=  num_rfcn_regions;
    }
  }

  template <typename Dtype>
  void RegionPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(mapping_channel_.count(), -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    BOXPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, num_features_, num_regions_,num_samples_, bottom[1]->shape(0), bottom[2]->gpu_data(), bottom_data, height_, width_,
      top_data, num_rfcn_regions_,mapping_channel_ptr,bottom[1]->gpu_data(),bottom[3]->gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void BOXPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_features,
    const int num_regions,
    const int num_samples,
    const int height, const int width, const int M,
    Dtype* bottom_diff, const int num_rfcn_regions,const Dtype* rfcn_regions_weights) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The input is in order (1, num_rfcn_regions * num_features , H, W ), nf = 64
      // sampled_points is in order (m, 2)
      // feat_ext_offset is in order (num_regions, num_samples, 2)
      // rfcn_region_idx
      // output is in order (m, num_regions, num_samples, num_features)
      // nthreads m * num_regions * num_features

      int w = index % width;
      int h = (index / width) % height;
      int f = (index / width / height) % num_features;
      int rfcn_r = index / width / height / num_features;
      for (int i = 0; i < M * num_regions * num_samples; i++ ){
        int s = i % num_samples;
        int r = (i / num_samples) % num_regions;
        int m = i / num_samples / num_regions;
        int top_idx = (((m * num_regions + r)* num_samples + s))*num_features + f ;
        int rfcn_regions_weights_idx = (((m * num_regions + r)* num_samples + s) * num_rfcn_regions + rfcn_r) ;
        if (mapping_channel[top_idx] == ((f * height + h) * width + w)){
          bottom_diff[index] += top_diff[top_idx] *rfcn_regions_weights[rfcn_regions_weights_idx];
        }

      }
      bottom_diff[index] /= num_rfcn_regions;
    }
  }

  template <typename Dtype>
  void RegionPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    BOXPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      num_features_, num_regions_,num_samples_,  height_, width_, bottom[1]->shape(0), bottom_diff,num_rfcn_regions_,bottom[3]->gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(RegionPoolingLayer);

}  // namespace caffe
