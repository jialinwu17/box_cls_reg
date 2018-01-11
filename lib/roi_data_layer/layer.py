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
from roi_data_layer.new_minibatch import get_minibatch
import numpy as np
import yaml
import scipy.io as sio
from multiprocessing import Process, Queue

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        num_regions = cfg.TRAIN.num_regions
        num_regions_one_side = np.sqrt(num_regions)
        num_samples = cfg.TRAIN.num_samples
        self._roidb = roidb
        self._shuffle_roidb_inds()
        self._sampled_id = np.zeros((num_regions*num_samples,2))
        self._bound_h = np.zeros((np.sqrt(num_regions)/2  + 1))
        self._bound_w = np.zeros((np.sqrt(num_regions)/2  + 1))
        wid = np.zeros((len(roidb)))
        hei = np.zeros((len(roidb)))
        for i in xrange(wid.shape[0]):
            wid[i] = roidb[i]['train_boxes_size'][0]
            hei[i] = roidb[i]['train_boxes_size'][1]
        sorted_w = np.sort(wid)
        sorted_h = np.sort(hei)
        for i in xrange( np.sqrt(num_regions)/2 ):
            self._bound_w[i+1] = sorted_w[ np.ceil(len(roidb)*(i+1) / (np.sqrt(num_regions)/2))]
            self._bound_h[i+1] = sorted_h[ np.ceil(len(roidb)*(i+1) / (np.sqrt(num_regions)/2))]
        sio.savemat('bound.mat',{'bound_w' = self._bound_w,'bound_h' = self._bound_h})
        
        for i in xrange(num_regions_one_side):
            for j in xrange(num_regions_one_side):
                if i < num_regions_one_side / 2:
                    bound_x1 = - self._bound_w[num_regions_one_side / 2 - i]
                    bonnd_x2 = - self._bound_w[num_regions_one_side / 2 - i - 1]
                else:
                    bound_x1 = self._bound_w[i - num_regions_one_side / 2]
                    bonnd_x2 = self._bound_w[i - num_regions_one_side / 2 + 1]
                if j < num_regions_one_side / 2:
                    bound_y1 = - self._bound_h[num_regions_one_side / 2 - j]
                    bonnd_y2 = - self._bound_h[num_regions_one_side / 2 - j - 1]
                else:
                    bound_y1 = self._bound_h[j - num_regions_one_side / 2]
                    bonnd_y2 = self._bound_h[j - num_regions_one_side / 2 + 1]
                bound_x1 = np.ceil(bound_x1)
                bound_y1 = np.ceil(bound_y1)
                bound_x2 = np.floor(bound_x2)
                bound_y2 = np.floor(bound_y2)
                x_step = (bound_x2 - bound_x1)/(np.sqrt(num_samples) )
                y_step = (bound_y2 - bound_y1)/(np.sqrt(num_samples) )
                self._sampled_id[(i* num_regions_one_side + j)*num_samples : (i* num_regions_one_side + j + 1)*num_samples ,0 ] = bound_x1 :bound_x2:x_step
                self._sampled_id[(i* num_regions_one_side + j)*num_samples : (i* num_regions_one_side + j + 1)*num_samples ,1 ] = bound_y1 :bound_y2:y_step
        self._sampled_id = np.around(self._sampled_id * cfg.TRAIN.spatial_scale)
        sio.savemat('sampled_id.mat',{'sampled_id' = self._sampled_id})
                


    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        self._num_regions = cfg.TRAIN.num_regions
        self._num_regions_one_side = np.sqrt(layer_params['num_regions'])
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5, 1, 1)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1, 1, 1, 1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                num_reg_class = 2 if cfg.TRAIN.AGNOSTIC else self._num_classes
                top[idx].reshape(1, num_reg_class * 4, 1, 1)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, num_reg_class * 4, 1, 1)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, num_reg_class * 4, 1, 1)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1
        top[idx].reshape( cfg.TRAIN.num_samples*cfg.TRAIN.num_regions , 2)
        self._name_to_top_map['sampled_id'] = idx
        idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            shape = blob.shape
            if len(shape) == 1:
                blob = blob.reshape(blob.shape[0], 1, 1, 1)
            if len(shape) == 2 and blob_name != 'im_info':
                blob = blob.reshape(blob.shape[0], blob.shape[1], 1, 1)
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
        top[len(blob)].reshape(*(self._sampled_id.shape))
        top[len(blob)].data[...] = self._sampled_id.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
