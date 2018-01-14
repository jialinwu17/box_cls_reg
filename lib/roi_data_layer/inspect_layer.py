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
import numpy as np
import yaml
from multiprocessing import Process, Queue
import scipy.io as sio
class InspectLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    
    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._str = layer_params['string']
        bottom_shape = bottom[0].shape
        top[0].reshape(*bottom_shape)
        

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        #print '%s forwarding'%self._str
        bottom_shape = bottom[0].shape
        #print bottom_shape[0]
        top[0].reshape(*bottom_shape)
        top[0].data[...] = bottom[0].data
        #print '%s forward,min:%.10f,max:%.10f,mean:%.10f,var:%.10f'%(self._str,top[0].data.min(),top[0].data.max(),top[0].data.mean(),top[0].data.var())
        #print 'shape:(%d,%d,%d,%d)'%(bottom[0].data.shape[0],bottom[0].data.shape[1],bottom[0].data.shape[2],bottom[0].data.shape[3])
        #sio.savemat(self._str + '.mat', {'data': bottom[0].data})
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        top_shape = top[0].shape
        bottom[0].diff[...] = top[0].diff[...]
        #sio.savemat(self._str + '.mat', {'data': bottom[0].data})
        print '%s backward,diff:%.10f'%(self._str,top[0].diff.mean())
        #print '%s backward,min:%.10f,max:%.10f,mean:%.10f,var:%.10f'%(self._str,top[0].diff.min(),top[0].diff.max(),top[0].diff.mean(),top[0].diff.var())

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

