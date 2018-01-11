# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import scipy.io as sio
class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    
    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        shape = bottom[0].data.shape #1, C, H, W

        self._num_region = layer_params['num_region']
        self._num_region_one_side = np.sqrt(self._num_region )
        self._num_classes = layer_params['num_classes']
        self._name_to_top_map = {}
        self._num_roi = bottom[1].data.shape[0]
        self._roi_scale = layer_params['roi_scale']
        self._region_bound = sio.loadmat('region_boundary.mat')

        
        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, (self._num_classes +1) , shape[2], shape[3])
        self._name_to_top_map['catetory_cls_labels'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes * self._num_region , shape[2], shape[3])
        self._name_to_top_map['bbox_cls_labels'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes * self._num_region_one_side*2 , shape[2], shape[3])
        self._name_to_top_map['bbox_reg_labels'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes   * self._num_region_one_side*2  , shape[2], shape[3])
        self._name_to_top_map['bbox_inweights'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, self._num_classes   * self._num_region_one_side*2  , shape[2], shape[3])
        self._name_to_top_map['bbox_outweights'] = idx
        idx += 1

        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        rois = bottom[1].data
        # bottom[0] 
        #
        # bottom[0] cls_score map (1, (K+1), H, W)
        cls_score = bottom[0].data
        catetory_cls_labels = np.zeros((cfg.TRAIN.IMS_PER_BATCH, self._num_classes + 1 , shape[2], shape[3]))
        bbox_cls_labels = np.zeros((cfg.TRAIN.IMS_PER_BATCH, 1 , shape[2], shape[3]))
        bbox_cls_inweights = np.zeros((cfg.TRAIN.IMS_PER_BATCH, self._num_classes * self._num_region , shape[2], shape[3]))
        
        bbox_reg_labels = np.zeros((cfg.TRAIN.IMS_PER_BATCH, self._num_classes * self._num_region_one_side * 2 , shape[2], shape[3]))
        bbox_inweights = np.zeros((cfg.TRAIN.IMS_PER_BATCH, self._num_classes * self._num_region_one_side * 2 , shape[2], shape[3]))
        bbox_outweights = np.ones((cfg.TRAIN.IMS_PER_BATCH, self._num_classes * self._num_region_one_side * 2,shape[2], shape[3]))
        
        for nroi in xrange(self._num_roi):
            count = 0
            roi_x1 =  rois[nroi,1] * self._roi_scale
            roi_y1 =  rois[nroi,2] * self._roi_scale
            roi_x2 =  rois[nroi,3] * self._roi_scale
            roi_y2 =  rois[nroi,4] * self._roi_scale
            center_w = (roi_x1 + roi_x2)/2
            center_h = (roi_y1 + roi_y2)/2
            area = (rois[nroi,4] - rois[nroi,2] + 1 )*(rois[nroi,3] - rois[nroi,1] + 1)
            score_in_roi = cls_score[0,1:,np.ceil(roi_y1):np.floor(roi_y2),np.ceil(roi_x1):np.floor(roi_x2)]
            max_score_in_roi = score_in_roi.max(axis = 1).reshape((shape[2]* shape[3]))
            sorted_score = np.sort(max_score_in_roi)
            
            for x in xrange(np.ceil(roi_x1),np.floor(roi_x2)):
                for y in xrange(np.ceil(roi_y1),np.floor(roi_y2)):
                    point2top = (y - roi_y1)/self._roi_scale
                    point2left = (x - roi_x1)/self._roi_scale
                    point2bottom =( - y + roi_y2)/self._roi_scale
                    point2right = (- x + roi_x2)/self._roi_scale
                    top_bound = 0
                    catetory_cls_labels[0,rois[nroi, 0], y,x] = 1
                    for i in xrange(self._num_region_one_side / 2 ): #bound begin from 0
                        if self._region_bound_x[i] <  point2left and point2left < self._region_bound_x[1 + i]:
                            x_ind = self._num_region_one_side / 2 - i - 1
                            for j in xrange(self._num_region_one_side / 2 ): 
                                if self._region_bound_y[j] <  point2top:
                                    y_ind = self._num_region_one_side / 2 - j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                                if self._region_bound_y[j] <  point2bottom:
                                    y_ind = self._num_region_one_side / 2 + j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                            bbox_reg_labels[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + x_ind   ,y,x] = (point2left - (self._region_bound_x[-1] -self._region_bound_x[1 + i] ))/(self._region_bound_x[1 + i] - self._region_bound_x[i]) - 0.5
                            bbox_inweights[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + x_ind   ,y,x]= 1
                        if self._region_bound_x[i] <  point2right and point2right < self._region_bound_x[1 + i]:
                            x_ind = self._num_region_one_side / 2 + i - 1
                            for j in xrange(self._num_region_one_side / 2 ): 
                                if self._region_bound_x[j] <  point2top:
                                    y_ind = self._num_region_one_side / 2 - j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                                if self._region_bound_x[j] <  point2bottom:
                                    y_ind = self._num_region_one_side / 2 + j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                            bbox_reg_labels[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + x_ind   ,y,x] =(point2right -  (self._region_bound_x[-1] -self._region_bound_x[1 + i] ))/(self._region_bound_x[1 + i] - self._region_bound_x[i]) - 0.5
                            bbox_inweights[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + x_ind   ,y,x]= 1
                    
                    for i in xrange(self._num_region_one_side / 2 ): #bound begin from 0
                        if self._region_bound_y[i] <  point2top and point2top < self._region_bound_y[1 + i]:
                            y_ind = self._num_region_one_side / 2 - i - 1
                            for j in xrange(self._num_region_one_side / 2 ): 
                                if self._region_bound_x[j] <  point2left:
                                    x_ind = self._num_region_one_side / 2 - j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                                if self._region_bound_x[j] <  point2bright:
                                    x_ind = self._num_region_one_side / 2 + j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                            bbox_reg_labels[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + self._num_region_one_side+ y_ind   ,y,x] = (point2top -  (self._region_bound_y[-1] -self._region_bound_y[1 + i] ))/(self._region_bound_y[1 + i] - self._region_bound_y[i]) - 0.5
                            bbox_inweights[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + self._num_region_one_side+y_ind   ,y,x]= 
                            
                        if self._region_bound_y[i] <  point2bottom and point2bottom < self._region_bound_y[1 + i]:
                            y_ind = self._num_region_one_side / 2 + i - 1
                            for j in xrange(self._num_region_one_side / 2 ): 
                                if self._region_bound_x[j] <  point2left:
                                    x_ind = self._num_region_one_side / 2 - j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                                if self._region_bound_x[j] <  point2bright:
                                    x_ind = self._num_region_one_side / 2 + j - 1
                                    bbox_cls_labels[0,(rois[nroi, 0] - 1) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                    count += 1
                            bbox_reg_labels[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + self._num_region_one_side+ y_ind   ,y,x] = (point2bottom -  (self._region_bound_y[-1] -self._region_bound_y[1 + i] ))/(self._region_bound_y[1 + i] - self._region_bound_y[i]) - 0.5
                            bbox_inweights[0,(rois[nroi, 0] - 1) *self._num_region_one_side * 2 + self._num_region_one_side+y_ind   ,y,x]= 1
            count -= 4
            
        
            top[0].reshape(*(catetory_cls_labels.shape))
            top[1].reshape(*(bbox_cls_labels.shape))
            top[2].reshape(*(bbox_reg_labels.shape))
            top[3].reshape(*(bbox_inweights.shape))
            top[4].reshape(*(bbox_outweights.shape))
            # Copy data into net's input blobs
            top[0].data[...] = catetory_cls_labels
            top[1].data[...] = bbox_cls_labels
            top[2].data[...] = bbox_reg_labels
            top[3].data[...] = bbox_inweights
            top[4].data[...] = bbox_outweights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
