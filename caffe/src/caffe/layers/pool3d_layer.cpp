/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */




#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/pool3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void Pooling3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "PoolingLayer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "PoolingLayer takes a single blob as output.";
  kernel_height_ = this->layer_param_.pooling3d_param().kernel_height();
  kernel_width_ = this->layer_param_.pooling3d_param().kernel_width();
  kernel_depth_ = this->layer_param_.pooling3d_param().kernel_depth();
  stride_ = this->layer_param_.pooling3d_param().stride();
  temporal_stride_ = this->layer_param_.pooling3d_param().temporal_stride();
  pad_ = this->layer_param_.pooling3d_param().pad();
  if (pad_ != 0) {
    CHECK_EQ(this->layer_param_.pooling3d_param().pool(),
             PoolingParameter_PoolMethod_AVE)
        << "Padding implemented only for average pooling.";
  }
  /*
  vector<int> pool_shape(5);
  pool_shape[0] = bottom[0]->shape(0);
  pool_shape[1] = 1;
  pool_shape[2] = bottom[0]->shape(1);
  pool_shape[3] = bottom[0]->shape(2);
  pool_shape[4] = bottom[0]->shape(3);
  bottom[0]->reshape(pool_shape)
  */
  
  
  
  
  channels_ = bottom[0]->shape(1);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_ - kernel_height_) / stride_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_ - kernel_width_) / stride_)) + 1;
      
  printf("%d,%d,%d,%f\n",length_,kernel_depth_,temporal_stride_,ceil(static_cast<float>(length_ - kernel_depth_) / temporal_stride_));
  pooled_length_ = static_cast<int>(ceil(static_cast<float>(length_ - kernel_depth_) / temporal_stride_)) + 1;

  vector<int> pool_shape(5);
  pool_shape[0] = bottom[0]->shape(0);
  pool_shape[1] = channels_;
  pool_shape[2] = pooled_length_;
  pool_shape[3] = pooled_height_;
  pool_shape[4] = pooled_width_;
  printf("%d,%d,%d,%d,%d\n",pool_shape[0],pool_shape[1],pool_shape[2],pool_shape[3],pool_shape[4]);
  top[0]->Reshape(pool_shape);

  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(pool_shape);
  }
}
template <typename Dtype>
void Pooling3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
  vector<int> pool_shape(5);
  /*pool_shape[0] = bottom[0]->shape(0);
  pool_shape[1] = 1;
  pool_shape[2] = bottom[0]->shape(1);
  pool_shape[3] = bottom[0]->shape(2);
  pool_shape[4] = bottom[0]->shape(3);
  bottom[0]->Reshape(pool_shape);*/
  
  channels_ = bottom[0]->shape(1);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_ - kernel_height_) / stride_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_ - kernel_width_) / stride_)) + 1;
  pooled_length_ = static_cast<int>(ceil(static_cast<float>(
	      length_ - kernel_depth_) / temporal_stride_)) + 1;

  //vector<int> pool_shape(5);
  pool_shape[0] = bottom[0]->shape(0);
  pool_shape[1] = channels_;
  pool_shape[2] = pooled_length_;
  pool_shape[3] = pooled_height_;
  pool_shape[4] = pooled_width_;
  
  

  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(pool_shape);
  }
  
  top[0]->Reshape(pool_shape);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  
}
template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	  const Dtype* bottom_data = bottom[0]->cpu_data();
	  Dtype* top_data = top[0]->mutable_cpu_data();
	  // Different pooling methods. We explicitly do the switch outside the for
	  // loop to save time, although this results in more codes.
	  int top_count = top[0]->count();
	  switch (this->layer_param_.pooling3d_param().pool()) {
	  case PoolingParameter_PoolMethod_MAX:
	    // Initialize
	    for (int i = 0; i < top_count; ++i) {
	      top_data[i] = -FLT_MAX;
	    }
	    // The main loop
	    for (int n = 0; n < bottom[0]->num(); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_;
	              int wstart = pw * stride_;
	              int lstart = pl * temporal_stride_;
	              int hend = min(hstart + kernel_height_, height_);
	              int wend = min(wstart + kernel_width_, width_);
	              int lend = min(lstart + kernel_depth_, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] =
	                      max(top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw],
	                          bottom_data[(l * height_ + h) * width_ + w]);
	                  }
	                }
	              }
	            }
	          }
	    	}
	        // compute offset
	        bottom_data += bottom[0]->offset(0, 1);
	        top_data += top[0]->offset(0, 1);
	      }
	    }
	    break;
	  case PoolingParameter_PoolMethod_AVE:
	    for (int i = 0; i < top_count; ++i) {
	      top_data[i] = 0;
	    }
	    // The main loop
	    for (int n = 0; n < bottom[0]->num(); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * temporal_stride_;
	              int hend = min(hstart + kernel_height_, height_ + pad_);
	              int wend = min(wstart + kernel_width_, width_ + pad_);
	              int lend = min(lstart + kernel_depth_, length_);
	              int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);

	              hend = min(hend, height_);
	              wend = min(wend, width_);
	              lend = min(lend, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] +=
	                        bottom_data[(l * height_ + h) * width_ + w];
	                  }
	                }
	              }
	              top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw] /= pool_size;
	            }
	          }
	    	}
	        // compute offset
	        bottom_data += bottom[0]->offset(0, 1);
	        top_data += top[0]->offset(0, 1);
	      }
	    }
	    break;
	  case PoolingParameter_PoolMethod_STOCHASTIC:
	    NOT_IMPLEMENTED;
	    break;
	  default:
	    LOG(FATAL) << "Unknown pooling method.";
	  }
	  

}

template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	  if (!propagate_down[0]) {
	    return;
	  }
	  const Dtype* top_diff = top[0]->cpu_diff();
	  const Dtype* top_data = top[0]->cpu_data();
	  const Dtype* bottom_data = bottom[0]->cpu_data();
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	  // Different pooling methods. We explicitly do the switch outside the for
	  // loop to save time, although this results in more codes.
	  memset(bottom_diff, 0, bottom[0]->count() * sizeof(Dtype));
	  switch (this->layer_param_.pooling3d_param().pool()) {
	  case PoolingParameter_PoolMethod_MAX:
	    // The main loop
	    for (int n = 0; n < top[0]->num(); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_;
	              int wstart = pw * stride_;
	              int lstart = pl * temporal_stride_;
	              int hend = min(hstart + kernel_height_, height_);
	              int wend = min(wstart + kernel_width_, width_);
	              int lend = min(lstart + kernel_depth_, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    bottom_diff[(l * height_ + h) * width_ + w] +=
	                        top_diff[(pl * pooled_height_ + ph) * pooled_width_ + pw] *
	                        (bottom_data[(l * height_ + h) * width_ + w] ==
	                            top_data[(pl * pooled_height_ + ph) * pooled_width_ + pw]);
	                  }
	                }
	              }
	            }
	          }
	    	}
	        // offset
	        bottom_data += bottom[0]->offset(0, 1);
	        top_data += top[0]->offset(0, 1);
	        bottom_diff += bottom[0]->offset(0, 1);
	        top_diff += top[0]->offset(0, 1);
	      }
	    }
	    break;
	  case PoolingParameter_PoolMethod_AVE:
	    // The main loop
	    for (int n = 0; n < top[0]->num(); ++n) {
	      for (int c = 0; c < channels_; ++c) {
	    	for (int pl = 0; pl < pooled_length_; ++pl) {
	          for (int ph = 0; ph < pooled_height_; ++ph) {
	            for (int pw = 0; pw < pooled_width_; ++pw) {
	              int hstart = ph * stride_ - pad_;
	              int wstart = pw * stride_ - pad_;
	              int lstart = pl * temporal_stride_;
	              int hend = min(hstart + kernel_height_, height_ + pad_);
	              int wend = min(wstart + kernel_width_, width_ + pad_);
	              int lend = min(lstart + kernel_depth_, length_);
	              int pool_size = (hend - hstart) * (wend - wstart) * (lend - lstart);
	              hstart = max(hstart, 0);
	              wstart = max(wstart, 0);
	              hend = min(hend, height_);
	              wend = min(wend, width_);
	              lend = min(lend, length_);
	              for (int l = lstart; l < lend; ++l) {
	                for (int h = hstart; h < hend; ++h) {
	                  for (int w = wstart; w < wend; ++w) {
	                    bottom_diff[(l * height_ + h) * width_ + w] +=
	                      top_diff[(pl * pooled_height_ + ph) * pooled_width_ + pw] / pool_size;
	                  }
	                }
	              }
	            }
	          }
	    	}
	        // offset
	        bottom_data += bottom[0]->offset(0, 1);
	        top_data += top[0]->offset(0, 1);
	        bottom_diff += bottom[0]->offset(0, 1);
	        top_diff += top[0]->offset(0, 1);
	      }
	    }
	    break;
	  case PoolingParameter_PoolMethod_STOCHASTIC:
	    NOT_IMPLEMENTED;
	    break;
	  default:
	    LOG(FATAL) << "Unknown pooling method.";
	  }

}


INSTANTIATE_CLASS(Pooling3DLayer);
REGISTER_LAYER_CLASS(Pooling3D);

}  // namespace caffe
