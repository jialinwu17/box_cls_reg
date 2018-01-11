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
import numpy as np
import yaml
from multiprocessing import Process, Queue
import scipy.io as sio
class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    
    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        # bottom 0 : rfcn confidence (1, 7, 7, 21, H, W)
        # bottom 1 : rois (num_rois, 5)
        # bottom 2 : feat_ext_offsets (num_regions * num_samples, 2)
        # top 0 : seed_points (num_rois * M, 2)
        # top 1 : rfcn_region_idx (num_rois * M, num_regions * num_samples, num_rfcn_regions) 
        # each values indicate the weights for the rfcn region 
        # top 2 bbox_cls_labels (num_rois * M, num_classes * num_regions)
        # top 3 bbox_reg_labels (num_rois * M, num_classes * num_regions_one_side * 2)


        self._num_regions = cfg.TRAIN.num_regions
        self._num_regions_one_side = np.sqrt(self._num_regions )
        self._num_classes = layer_params['num_classes']
        self._agnostic_box = layer_params['agnostic_box']
        if self._agnostic_box :
            self._num_classes_box = 1
        else:
            self._num_classes_box = self._num_classes
        self._name_to_top_map = {}
        self._num_rois = bottom[1].data.shape[0]
        self._roi_scale = layer_params['roi_scale']
        self._region_bound = sio.loadmat('bound.mat')
        self._region_bound_x = self._region_bound['bound_w']
        self._region_bound_y = self._region_bound['bound_h']

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape( self._num_rois * cfg.TRAIN.M, 2)
        self._name_to_top_map['seed_points'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, self._num_regions * cfg.TRAIN.num_samples, cfg.TRAIN.num_rfcn_regions)
        self._name_to_top_map['rfcn_region_weights'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, self._num_classes * self._num_regions)
        self._name_to_top_map['bbox_cls_labels'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, , self._num_classes * self._num_regions_one_side * 2)
        self._name_to_top_map['bbox_cls_inweights'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, , self._num_classes * self._num_regions_one_side * 2)
        self._name_to_top_map['bbox_cls_outweights'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, , self._num_classes * self._num_regions_one_side * 2)
        self._name_to_top_map['bbox_reg_labels'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, , self._num_classes * self._num_regions_one_side * 2)
        self._name_to_top_map['bbox_reg_inweights'] = idx
        idx += 1

        top[idx].reshape(self._num_rois * cfg.TRAIN.M, , self._num_classes * self._num_regions_one_side * 2)
        self._name_to_top_map['bbox_reg_outweights'] = idx
        idx += 1
        # only regress class specific weights and classifier



    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        rois = bottom[1].data
        rfcn_feat = bottom[3].data
        feat_ext_offsets = bottom[2].data
        # parse the layer parameter string, which must be valid YAML
        # bottom 0 : rfcn confidence (1, 7, 7, 21, H, W)
        # bottom 1 : rois (num_rois, 5)
        # bottom 2 : feat_ext_offsets (num_regions * num_samples, 2)
        # bottom 3 : rfcn features
        # top 0 : seed_points (num_rois * M, 2)
        # top 1 : rfcn_region_weights (num_rois * M, num_regions * num_samples, num_rfcn_regions) 
        # each values indicate the weights for the rfcn region 
        # top 2 bbox_cls_labels (num_rois * M, num_classes * num_regions)
        # top 3 bbox_reg_labels (num_rois * M, num_classes * num_regions_one_side * 2)

        rfcn_conf = bottom[0].data
        seed_points = np.zeros((self._num_rois * cfg.TRAIN.M, 2))
        seed_points_feat = np.zeros((self._num_rois * cfg.TRAIN.M, cfg.TRAIN.num_features))
        rfcn_region_weights = np.zeros((self._num_rois * cfg.TRAIN.M,cfg.TRAIN.num_regions*cfg.TRAIN.num_samples, cfg.TRAIN.num_rfcn_regions))
        bbox_cls_labels = np.zeros((self._num_rois * cfg.TRAIN.M, self._num_classes_box * cfg.TRAIN.num_regions))
        bbox_reg_labels = np.zeros((self._num_rois * cfg.TRAIN.M, self._num_classes_box * self._num_regions_one_side * 2))

        
        bbox_cls_inweights = np.zeros((self._num_rois * cfg.TRAIN.M, self._num_classes_box *  cfg.TRAIN.num_regions))
        bbox_cls_outweights = np.ones((self._num_rois * cfg.TRAIN.M, self._num_classes_box *  cfg.TRAIN.num_regions))
        
        bbox_inweights = np.zeros((self._num_rois * cfg.TRAIN.M, self._num_classes_box * self._num_regions_one_side * 2 ))
        bbox_outweights = np.ones((self._num_rois * cfg.TRAIN.M, self._num_classes_box * self._num_regions_one_side * 2))
        
        for nroi in xrange(self._num_roi):
            count = 0
            roi_x1 =  rois[nroi,1] * self._roi_scale
            roi_y1 =  rois[nroi,2] * self._roi_scale
            roi_x2 =  rois[nroi,3] * self._roi_scale
            roi_y2 =  rois[nroi,4] * self._roi_scale
            center_w = (roi_x1 + roi_x2)/2
            center_h = (roi_y1 + roi_y2)/2
            #area = (rois[nroi,4] - rois[nroi,2] + 1 )*(rois[nroi,3] - rois[nroi,1] + 1)
            roi_class = rois[nroi,0]
            rfcn_conf_in_roi = rfcn_conf[0,:,:,roi_class,np.ceil(roi_y1):np.floor(roi_y2),np.ceil(roi_x1):np.floor(roi_x2)]
            rfcn_conf_in_roi = rfcn_conf_in_roi.reshape((cfg.TRAIN.num_rfcn_regions,rfcn_conf_in_roi.shape[2],rfcn_conf_in_roi.shape[3] ))
            #shape (49,h,w)
            max_class_conf_in_roi = (np.max(rfcn_conf_in_roi,axis = 0)).reshape((rfcn_conf_in_roi.shape[1]*rfcn_conf_in_roi.shape[2]))
            seed_points_inds = np.argsort(max_score_in_roi)[ - cfg.TRAIN.M :]
            for m in xrange(cfg.TRAIN.M):
                seed_x_in_roi = seed_points_inds[m] % rfcn_conf_in_roi.shape[2]
                seed_y_in_roi = seed_points_inds[m] / rfcn_conf_in_roi.shape[2]
                seed_x = (seed_x_in_roi + np.ceil(roi_x1)) 
                seed_y = (seed_y_in_roi + np.ceil(roi_y1)) 
                seed_rfcn_region = np.argmax(rfcn_conf_in_roi[:,seed_y_in_roi,seed_x_in_roi ])[0]
                region_x = seed_rfcn_region % (np.sqrt(cfg.TRAIN.num_rfcn_regions))
                region_y = seed_rfcn_region / (np.sqrt(cfg.TRAIN.num_rfcn_regions))
                seed_points[nroi * cfg.TRAIN.M + m,0] = seed_x
                seed_points[nroi * cfg.TRAIN.M + m,1] = seed_y
                seed_points_feat[nroi * cfg.TRAIN.M + m,:] = rfcn_feat[0,seed_rfcn_region*cfg.TRAIN.num_features:(seed_rfcn_region+1)*cfg.TRAIN.num_features, seed_y,seed_x]
                for idx in xrange(cfg.TRAIN.num_regions * cfg.TRAIN.num_samples):
                    sampled_x = seed_x + feat_ext_offsets[idx , 0]
                    sampled_y = seed_y + feat_ext_offsets[idx , 1]
                    if sampled_x >=0 and sampled_x < rfcn_conf.shape[5] and sampled_y >= 0 and sampled_y < rfcn_conf.shape[4] :
                        region_weights = rfcn_conf[0,:,:,roi_class,sampled_y,sampled_x]
                        if feat_ext_offsets[idx , 0] > 0 : 
                            region_weights[:,0:seed_x_in_roi] = 0.0
                        else:
                            region_weights[:,seed_x_in_roi + 1 : ] = 0.0
                        if feat_ext_offsets[idx , 1] > 0 : 
                            region_weights[0:seed_y_in_roi,:] = 0.0
                        else:
                            region_weights[seed_y_in_roi + 1:,:] = 0.0
                        rfcn_region_weights[nroi * cfg.TRAIN.M + m, idx, :] = region_weights.reshape((cfg.TRAIN.num_rfcn_regions))

                if self._agnostic_box:
                    box_category_label = 0 - np.minimum(0,rois[nroi,0])
                else:
                    box_category_label = rois[nroi,0] - 1
            
                point2top = (seed_y - roi_y1)/self._roi_scale
                point2left = (seed_x - roi_x1)/self._roi_scale
                point2bottom =( - seed_y + roi_y2)/self._roi_scale
                point2right = (- seed_x + roi_x2)/self._roi_scale
                top_bound = 0

                
                bbox_cls_inweights[ nroi * cfg.TRAIN.M + m ,box_category_label * self._num_region: (box_category_label+1) * self._num_region] =  1 / cfg.TRAIN.M

                for i in xrange(self._num_region_one_side / 2 ): #bound begin from 0
                    if self._region_bound_x[i] <  point2left and point2left < self._region_bound_x[1 + i]:
                        x_ind = self._num_region_one_side / 2 - i - 1
                        for j in xrange(self._num_region_one_side / 2 ): 
                            if self._region_bound_y[j] <  point2top:
                                y_ind = self._num_region_one_side / 2 - j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) * self._num_region + y_ind * self._num_region_one_side + x_ind] = 1
                                count += 1
                            if self._region_bound_y[j] <  point2bottom:
                                y_ind = self._num_region_one_side / 2 + j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) * self._num_region + y_ind * self._num_region_one_side + x_ind ] = 1
                                count += 1
                        bbox_reg_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) *self._num_region_one_side * 2 + x_ind ] = ( self._region_bound_x[1 + i] - point2left)/(self._region_bound_x[1 + i] - self._region_bound_x[i]) - 0.5
                        bbox_inweights[nroi * cfg.TRAIN.M + m ,(box_category_label) *self._num_region_one_side * 2 + x_ind ]= 1
                    if self._region_bound_x[i] <  point2right and point2right < self._region_bound_x[1 + i]:
                        x_ind = self._num_region_one_side / 2 + i - 1
                        for j in xrange(self._num_region_one_side / 2 ): 
                            if self._region_bound_x[j] <  point2top:
                                y_ind = self._num_region_one_side / 2 - j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                count += 1
                            if self._region_bound_x[j] <  point2bottom:
                                y_ind = self._num_region_one_side / 2 + j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                count += 1
                        bbox_reg_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) *self._num_region_one_side * 2 + x_ind ] = ( self._region_bound_x[1 + i] - point2right)/(self._region_bound_x[1 + i] - self._region_bound_x[i]) - 0.5
                        bbox_inweights[nroi * cfg.TRAIN.M + m ,(box_category_label) *self._num_region_one_side * 2 + x_ind ]= 1
                
                for i in xrange(self._num_region_one_side / 2 ): #bound begin from 0
                    if self._region_bound_y[i] <  point2top and point2top < self._region_bound_y[1 + i]:
                        y_ind = self._num_region_one_side / 2 - i - 1
                        for j in xrange(self._num_region_one_side / 2 ): 
                            if self._region_bound_x[j] <  point2left:
                                x_ind = self._num_region_one_side / 2 - j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,(box_category_label) * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                count += 1
                            if self._region_bound_x[j] <  point2bright:
                                x_ind = self._num_region_one_side / 2 + j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,box_category_label * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                count += 1
                        bbox_reg_labels[nroi * cfg.TRAIN.M + m ,box_category_label*self._num_region_one_side * 2 + self._num_region_one_side+ y_ind ] = ( self._region_bound_y[1 + i] - point2top )/(self._region_bound_y[1 + i] - self._region_bound_y[i]) - 0.5
                        bbox_inweights[nroi * cfg.TRAIN.M + m ,box_category_label *self._num_region_one_side * 2 + self._num_region_one_side+y_ind ]= 1
                            
                    if self._region_bound_y[i] <  point2bottom and point2bottom < self._region_bound_y[1 + i]:
                        y_ind = self._num_region_one_side / 2 + i - 1
                        for j in xrange(self._num_region_one_side / 2 ): 
                            if self._region_bound_x[j] <  point2left:
                                x_ind = self._num_region_one_side / 2 - j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,box_category_label * self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                count += 1
                            if self._region_bound_x[j] <  point2bright:
                                x_ind = self._num_region_one_side / 2 + j - 1
                                bbox_cls_labels[nroi * cfg.TRAIN.M + m ,box_category_label* self._num_region + y_ind * self._num_region_one_side + x_ind ,y,x] = 1
                                count += 1
                        bbox_reg_labels[nroi * cfg.TRAIN.M + m ,box_category_label*self._num_region_one_side * 2 + self._num_region_one_side+ y_ind ] = ( self._region_bound_y[1 + i] -point2bottom )/(self._region_bound_y[1 + i] - self._region_bound_y[i]) - 0.5
                        bbox_inweights[nroi * cfg.TRAIN.M + m ,box_category_label*self._num_region_one_side * 2 + self._num_region_one_side+y_ind ]= 1
                count -= 4
            
        
            top[0].reshape(*(seed_points.shape))
            top[1].reshape(*(rfcn_region_weights.shape))
            top[2].reshape(*(bbox_cls_labels.shape))
            top[3].reshape(*(bbox_cls_inweights.shape))
            top[4].reshape(*(bbox_cls_outweights.shape))
            top[5].reshape(*(bbox_reg_labels.shape))
            top[6].reshape(*(bbox_inweights.shape))
            top[7].reshape(*(bbox_outweights.shape))
            top[8].reshape(*(seed_points_feat.shape))
            # Copy data into net's input blobs
            top[0].data[...] = seed_points
            top[1].data[...] = rfcn_region_weights
            top[2].data[...] = bbox_cls_labels
            top[3].data[...] = bbox_cls_inweights
            top[4].data[...] = bbox_cls_outweights
            top[5].data[...] = bbox_reg_labels
            top[6].data[...] = bbox_inweights
            top[7].data[...] = bbox_outweights
            top[8].data[...] = seed_points_feat.shape

        



    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass
        

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
