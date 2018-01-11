#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>
#include "caffe/layers/box_pooling_layer.hpp"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void BOXPoolingForward(
    const int nthreads,
    const int num_features, const int num_regions,const int num_samples,
    const Dtype* bottom_data,
    const int height, const int width,
    Dtype* top_data,
    const int num_rfcn_regions,
    int* mapping_channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The input is in order (1, nr*nf , H, W), nf = 64
      
      int output_dim = num_features * num_regions;
      int w = index % width;
      int h = (index / width) % height;
      int ctop = (index / width / height) % output_dim; // outdim = N_feature* n_region
      int feature = (index / width / height) % num_features;
      int region = (index / width / height/ num_features) % num_regions; 
      int n = index / width / height / output_dim;
      int direction = 4 * ctop / output_dim;
      int num_regions_one_side = (int)(sqrt(num_regions))
      int num_rfcn_regions_one_side = (int)(sqrt(num_rfcn_regions))
      int w_region = region% num_regions_one_side;
      int h_region = region/ num_regions_one_side;
      float centered_w_region = (float)w_region - (float)num_regions_one_side / 2.0)
      float centered_h_region = (float)h_region - (float)num_regions_one_side / 2.0)

      for (int i = 0;i< num_samples; i++){
        int x = sample_id[region*num_samples + i];
        int y = sample_id[num_regions* num_samples + region*num_samples + i];
        int sampled_x = w + x;
        int sampled_y = h + y;

        if (sampled_x >= 0 && sampled_x < width)&&(sampled_y >= 0 && sampled_y < height){
          
          int tmp[num_rfcn_regions] = {0};
          for (int j = 0; j< num_rfcn_region; j ++ ){
            
            int w_rfcn_region = j % num_rfcn_regions_one_side;
            int h_rfcn_region = j / num_rfcn_regions_one_side;
            float centered_w_rfcn_region = (float)w_rfcn_region - (float)num_rfcn_regions_one_side / 2.0)
            float centered_h_rfcn_region = (float)h_rfcn_region - (float)num_rfcn_regions_one_side / 2.0)
            if ((centered_w_rfcn_region* centered_w_region >0)&& (centered_h_rfcn_region* centered_h_region >0)){
              int bottom_idx = n*output_dim*height*width + j*num_features*height*width + sampled_y * width + sampled_x;
              for (int k = 0;k<num_features;k++){
                  tmp[j] += bottom_data[bottom_idx + k]*bottom_data[bottom_idx + k];
              }

            }
          }
          int selected = 0; int max_norm = 0;
          for (int j = 0; j< num_rfcn_region; j ++ ){
            if (tmp[j]> max_norm){
              max_norm = tmp[j];
              selected = j;
            }

          }
        
          int bottom_idx = n*output_dim*height*width + selected*num_features*height*width + feature * height*width + sampled_y * width + sampled_x;
          top_data[index] += bottom_data[bottom_idx];
          mapping_channel[ index*num_samples + i ] = index;
          mapping_channel[ nthreads*num_samples + index * num_samples + i ] = bottom_idx;
          // TODO: mapping back require search from the second half of mapping channels 

        }
        
      }
    }
  }

  template <typename Dtype>
  void BOXPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(mapping_channel_.count(), -1, mapping_channel_ptr);
    // NOLINT_NEXT_LINE(whitespace/operators)
    BOXPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, num_features_, num_regions_,num_samples_,bottom_data, height_, width_,
      top_data, num_rfcn_regions_,mapping_channel_ptr);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void BOXPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_features,
    const Dtype num_regions,
    const int num_samples,
    const int height, const int width,
    Dtype* bottom_diff,) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int output_dim = num_features * num_regions;
      int w = index % width;
      int h = (index / width) % height;
      int ctop = (index / width / height) % output_dim; // outdim = N_feature* n_region
      int feature = (index / width / height) % num_features;
      int region = (index / width / height/ num_features) % num_regions; 
      int n = index / width / height / output_dim;
      int direction = 4 * ctop / output_dim;
      
      for(int i = 0; i < nthreads * num_samples; i++ ){
        if ( mapping_channel[i + nthreads * num_samples ]== index){
          bottom_diff[ index ] += top_diff[mapping_channel[i]];
        } 
      }
    }
  }

  template <typename Dtype>
  void BOXPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
      num_features_, num_regions_,num_samples_,  height_, width_, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(BOXPoolingLayer);

}  // namespace caffe
