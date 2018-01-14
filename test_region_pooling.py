import numpy as np 
import scipy.io as sio
import sys
sys.path.append('caffe/python')
import caffe
proto = 'test_region_pooling.prototxt'
rfcn_feat = np.random.rand(1, 9 * 4, 10 , 10 )
seed_point = np.zeros((1,2))
seed_point[0,0] = 10
seed_point[0,1] = 10
sampled_id = sio.loadmat('sampled_id_1.mat')['sampled_id']
sampled_id = np.floor(sampled_id / 4.0)
rfcn_region_weights = np.ones((1, 576, 9 )) 

net = caffe.Net(proto, caffe.TRAIN)
caffe.set_device(0)
caffe.set_mode_gpu()
error = 0 
for i1 in xrange( 9* 4) :
	for i2 in xrange(10) :
		for i3 in xrange(10) :
			rfcn_feat = np.random.rand(1, 9 * 4, 10, 10)
			net.blobs['rfcn_features'].data[...] = rfcn_feat
			net.blobs['seed_points'].data[...] = seed_point
			net.blobs['sampled_id'].data[...] = sampled_id
			net.blobs['rfcn_region_weights'].data[...] = rfcn_region_weights
			out = net.forward()
			weighted_box_cls_1 = net.blobs['weighted_box_cls_1'].data
			rfcn_feat_new = rfcn_feat
			rfcn_feat_new[0,i1,i2,i3] += 0.0001
			net.blobs['rfcn_features'].data[...] = rfcn_feat
			net.blobs['seed_points'].data[...] = seed_point
			net.blobs['sampled_id'].data[...] = sampled_id
			net.blobs['rfcn_region_weights'].data[...] = rfcn_region_weights
			out = net.forward()
			weighted_box_cls_1_new = net.blobs['weighted_box_cls_1'].data
			for o1 in xrange(64):
				for o2 in xrange(9):
					for o3 in xrange(4):
						dy = weighted_box_cls_1_new[0,o1,o2,o3] - weighted_box_cls_1[0,o1,o2,o3]
						dx = 0.0001
						net.blobs['weighted_box_cls_1'].diff[...] = np.zeros((1,64,9,4))
						net.blobs['weighted_box_cls_1'].diff[0,o1,o2,o3] = 1.0
						d_est = dy / dx
						out = net.backward()
						d_com = net.blobs['rfcn_features'].diff[0,i1,i2,i3] 
						error += np.abs(d_com - d_est)

	print 'a channel'
	print error







