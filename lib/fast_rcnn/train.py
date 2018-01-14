# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os
import scipy.io as sio
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import scipy.io as sio
import google.protobuf.text_format
class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        num_regions = cfg.TRAIN.num_regions
        num_regions_one_side = int(np.sqrt(num_regions))
        num_samples = cfg.TRAIN.num_samples
        sampled_id = np.zeros((num_regions*num_samples,2))
        bound_h = np.zeros((np.sqrt(num_regions)/2  + 1))
        bound_w = np.zeros((np.sqrt(num_regions)/2  + 1))
        wid = {}
        hei = {}
        sorted_w = {}
        sorted_h = {}
        for i in xrange(1, cfg.TRAIN.num_classes):
            wid[str(i )] = []
            hei[str(i )] = []
        for i in xrange(len(roidb)/2):
            for j in xrange(roidb[i]['train_boxes_size'].shape[0]) :
                gt_classes = roidb[i]['gt_classes'][j]
                wid[str(gt_classes)].extend([roidb[i]['train_boxes_size'][j,0]])
                hei[str(gt_classes)].extend([roidb[i]['train_boxes_size'][j,1]])
        for k in xrange(1, cfg.TRAIN.num_classes):
            wid[str(k)] = np.array(wid[str(k)])
            hei[str(k)] = np.array(hei[str(k)])
            sorted_w[str(k)] = np.sort(wid[str(k)])
            sorted_h[str(k)] = np.sort(hei[str(k)])
            for i in xrange( 1,int(np.sqrt(num_regions)/2) +1 ):
                bound_w[i] = sorted_w[str(k)][ np.maximum(0, np.ceil(wid[str(k)].shape[0]*(i) / (np.sqrt(num_regions)/2)) - 1 )]
                bound_h[i] = sorted_h[str(k)][ np.maximum(0, np.ceil(wid[str(k)].shape[0]*(i) / (np.sqrt(num_regions)/2)) - 1 )]
            sio.savemat('bound_%d.mat'%(k),{'bound_w' : bound_w,'bound_h' : bound_h})
        
            for i in xrange(num_regions_one_side):
                for j in xrange(num_regions_one_side):
                    if i < num_regions_one_side / 2:
                        bound_x1 = - bound_w[num_regions_one_side / 2 - i]
                        bound_x2 = - bound_w[num_regions_one_side / 2 - i - 1]
                    else:
                        bound_x1 = bound_w[i - num_regions_one_side / 2]
                        bound_x2 = bound_w[i - num_regions_one_side / 2 + 1]
                    if j < num_regions_one_side / 2:
                        bound_y1 = - bound_h[num_regions_one_side / 2 - j]
                        bound_y2 = - bound_h[num_regions_one_side / 2 - j - 1]
                    else:
                        bound_y1 = bound_h[j - num_regions_one_side / 2]
                        bound_y2 = bound_h[j - num_regions_one_side / 2 + 1]
                    bound_x1 = np.ceil(bound_x1)
                    bound_y1 = np.ceil(bound_y1)
                    bound_x2 = np.floor(bound_x2)
                    bound_y2 = np.floor(bound_y2)
                    x_step = (bound_x2 - bound_x1)/(np.sqrt(num_samples) )
                    y_step = (bound_y2 - bound_y1)/(np.sqrt(num_samples) )
                    sampled_id[(i* num_regions_one_side + j)*num_samples : (i* num_regions_one_side + j + 1)*num_samples ,0 ] = np.tile(np.linspace(bound_x1,bound_x2,np.sqrt(num_samples) + 2 )[1:-1].reshape(1,np.sqrt(num_samples)),[np.sqrt(num_samples),1]).reshape((num_samples))
                    sampled_id[(i* num_regions_one_side + j)*num_samples : (i* num_regions_one_side + j + 1)*num_samples ,1 ] = np.tile(np.linspace(bound_y1,bound_y2,np.sqrt(num_samples) + 2 )[1:-1].reshape(np.sqrt(num_samples),1),[1,np.sqrt(num_samples)]).reshape((num_samples))
            sampled_id = np.around(sampled_id * cfg.TRAIN.spatial_scale)
            sio.savemat('sampled_id_%d.mat'%(k),{'sampled_id' : sampled_id})



        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self, model_name):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params_faster_rcnn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        scale_bbox_params_rfcn = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('rfcn_bbox'))

        scale_bbox_params_rpn = (cfg.TRAIN.RPN_NORMALIZE_TARGETS and
                                 net.params.has_key('rpn_bbox_pred'))

        if scale_bbox_params_faster_rcnn:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if scale_bbox_params_rpn:
            rpn_orig_0 = net.params['rpn_bbox_pred'][0].data.copy()
            rpn_orig_1 = net.params['rpn_bbox_pred'][1].data.copy()
            num_anchor = rpn_orig_0.shape[0] / 4
            # scale and shift with bbox reg unnormalization; then save snapshot
            self.rpn_means = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_MEANS),
                                      num_anchor)
            self.rpn_stds = np.tile(np.asarray(cfg.TRAIN.RPN_NORMALIZE_STDS),
                                     num_anchor)
            net.params['rpn_bbox_pred'][0].data[...] = \
                (net.params['rpn_bbox_pred'][0].data *
                 self.rpn_stds[:, np.newaxis, np.newaxis, np.newaxis])
            net.params['rpn_bbox_pred'][1].data[...] = \
                (net.params['rpn_bbox_pred'][1].data *
                 self.rpn_stds + self.rpn_means)

        if scale_bbox_params_rfcn:
            # save original values
            orig_0 = net.params['rfcn_bbox'][0].data.copy()
            orig_1 = net.params['rfcn_bbox'][1].data.copy()
            repeat = orig_1.shape[0] / self.bbox_means.shape[0]

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['rfcn_bbox'][0].data[...] = \
                    (net.params['rfcn_bbox'][0].data *
                     np.repeat(self.bbox_stds, repeat).reshape((orig_1.shape[0], 1, 1, 1)))
            net.params['rfcn_bbox'][1].data[...] = \
                    (net.params['rfcn_bbox'][1].data *
                     np.repeat(self.bbox_stds, repeat) + np.repeat(self.bbox_means, repeat))

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        #filename = os.path.join(self.output_dir, filename)
        filename = 'models/pascal_voc/ResNet-50/' + model_name +'/' + filename
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params_faster_rcnn:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        if scale_bbox_params_rfcn:
            # restore net to original state
            net.params['rfcn_bbox'][0].data[...] = orig_0
            net.params['rfcn_bbox'][1].data[...] = orig_1
        if scale_bbox_params_rpn:
            # restore net to original state
            net.params['rpn_bbox_pred'][0].data[...] = rpn_orig_0
            net.params['rpn_bbox_pred'][1].data[...] = rpn_orig_1

        return filename

    def train_model(self, max_iters,model_name):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            solver = self.solver
            net = solver.net
            keys = net.blobs.keys()
            saved_dict = {}
            if self.solver.iter % cfg.TRAIN.save_feat == 0:
                for k in keys :
                    if 'res' not in k:
                        saved_dict[k] = net.blobs[k].data

                sio.savemat('models/pascal_voc/ResNet-50/' + model_name +'/' + 'saved_dict_%d.mat'%self.solver.iter, saved_dict)






            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot(model_name)

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot(model_name))
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir,model_name,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters, model_name)
    print 'done solving'
    return model_paths
