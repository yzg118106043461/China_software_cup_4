# -*- coding: UTF-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2,csv,codecs
import argparse
import pandas as pd

CLASSES = ('__background__',
           '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_30000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'ResNet-50':('ResNet-50','ResNet-50_faster_rcnn_iter_100000.caffemodel'),
        'ResNet-101':('ResNet-101','resnet101_faster_rcnn_bn_scale_merged_end2end_iter_90000.caffemodel')
        }


def ab(dataframe):
	return '\t'.join(dataframe.values.tostring())


def vis_detections(image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
 
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
    if(class_name == '__background__'):
        fw = open('./result.txt','a')   #保存结果的文件，下同
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '1'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '2'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '3'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '4'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '5'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '6'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '7'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '8'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '9'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '10'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '11'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '12'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '13'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '14'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '15'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '16'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '17'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '18'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '19'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '20'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '21'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '22'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '23'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '24'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '25'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '26'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '27'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '28'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    elif(class_name == '29'):
        fw = open('./result.txt','a')
        fw.write(str(image_name)+' '+class_name+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()
    """df = pd.read_csv("result.csv",sep='\t')
    df=df.groupby(['image_name'])['location','class_name'].apply(ab)
    df = df.reset_index()"""
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]
    fig,ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(image_name, cls, dets, thresh=CONF_THRESH)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-50]',
                        choices=NETS.keys(), default='ResNet-50')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)


    #df = pd.read_csv(r"result.csv",converters={u'image_name':str})
    

    print '\n\nLoaded network {:s}'.format(caffemodel)



    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = os.listdir("./data/demo")
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
        #plt.savefig("./data/testfig/" + im_name, format = 'jpg',transparent = True,pad_inches = 0,dpi = 300,bbox_inches = 'tight')
