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

        self._tiles = layer_params['tiles']
        self._axis = layer_params['axis']



    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        
        data = bottom[0].data
        data = data[:,,:,:]
        data = np.transpose(data,self._order) 
        top[0].reshape(*(data.shape))
        top[0].data[...] = data


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        diff = top[0].diff
        diff = np.transpose(diff, self._order_back)
        bottom[0].reshape((*diff.shape))
        bottom[0].diff = diff


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
