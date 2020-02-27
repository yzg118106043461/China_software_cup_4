
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
           'chinese','party','army','usa','britain','france','japan','northkorea','southkorea',
                         'russia','spain','olympic','un','eu','philippines','india','brazil','cambodia','laos',
                         'vietnam','malaysia','myanmar','singapore','thailand','afghanistan','iraq','iran','syria',
                         'jordan','lebanon','israel','palestine','australian','canada','saudiarabia','sweden',
                         'asean','belarus','nato','wto')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_40000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'ResNet-50':('ResNet-50','ResNet-50_faster_rcnn_iter_45000.caffemodel'),
        'ResNet-101':('ResNet-101','resnet101_faster_rcnn_bn_scale_merged_end2end_iter_70000.caffemodel')
        }

def ab(dataframe):
	return '\t'.join(dataframe.values.tostring())

def vis_detections(image_name, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    data = pd.DataFrame(columns=['image_name','location','class_name'])
    if len(inds) == 0:
        return

#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        

        """ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1) 
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')"""

        if class_name == 'chinese':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(1)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'party':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(2)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'army':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(3)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'usa':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(4)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'britain':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(5)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'france':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(6)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'japan':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(7)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'northkorea':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(8)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'southkorea':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(9)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'russia':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(10)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'spain':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(11)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'olympic':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(12)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'un':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(13)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'eu':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(14)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'philippines':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(16)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'india':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(16)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'brazil':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(17)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'cambodia':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(18)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'laos':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(19)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'vietnam':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(20)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'malaysia':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(21)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'myanmar':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(22)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'singapore':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(23)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'thailand':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(24)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'afghanistan':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(25)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'iraq':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(26)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'iran':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(27)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'syria':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(28)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'jordan':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(29)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'lebanon':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(30)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'israel':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(31)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'palestine':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(32)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'australian':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(33)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'canada':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(34)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'saudiarabia':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(35)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'sweden':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(36)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'asean':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(37)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'belarus':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(38)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'nato':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(39)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        elif class_name == 'wto':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
        	insertRow = pd.DataFrame({'image_name':image_name,'location':location,'class_name':str(40)},columns=['image_name','location','class_name'])
        	data = data.append(insertRow,ignore_index=True)
        

    		

    	"""elif class_name == 'army':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
    		data = {'image_name':image_name,'location':location,'class_name':3}
    		dataframe = pd.DataFrame(data,columns=['image_name','location','class_name'],index=[i])
    		dataframe.to_csv("result.csv",header=False,index=False,sep='\t',mode='a')
    	elif class_name == 'usa':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
    		data = {'image_name':image_name,'location':location,'class_name':4}
    		dataframe = pd.DataFrame(data,columns=['image_name','location','class_name'],index=[i])
    		dataframe.to_csv("result.csv",header=False,index=False,sep='\t',mode='a')
    	elif class_name == 'britain':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
    		data = {'image_name':image_name,'location':location,'class_name':5}
    		dataframe = pd.DataFrame(data,columns=['image_name','location','class_name'],index=[i])
    		dataframe.to_csv("result.csv",header=False,index=False,sep='\t',mode='a')
    	elif class_name == 'france':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
    		data = {'image_name':image_name,'location':location,'class_name':6}
    		dataframe = pd.DataFrame(data,columns=['image_name','location','class_name'],index=[0])
    		dataframe.to_csv("result.csv",header=False,index=False,sep='\t',mode='a')
    	elif class_name == 'japan':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
    		data = {'image_name':image_name,'location':location,'class_name':7}
    		dataframe = pd.DataFrame(data,columns=['image_name','location','class_name'],index=[0])
    		dataframe.to_csv("result.csv",header=False,index=False,sep='\t',mode='a')
    	elif class_name == 'northkorea':
        	location = []
        	location.append((bbox[0],bbox[1],bbox[2] - bbox[0],bbox[3] - bbox[1]))
    		data = {'image_name':image_name,'location':location,'class_name':8}
    		dataframe = pd.DataFrame(data,columns=['image_name','location','class_name'],index=[0])
    		dataframe.to_csv("result.csv",header=False,index=False,sep='\t',mode='a')"""
    data.to_csv("result2.csv",header=False,index=False,sep='\t',mode='a')
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
        vis_detections(image_name, cls, dets,ax, thresh=CONF_THRESH)
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
        plt.savefig("./data/testfig/" + im_name, format = 'jpg',transparent = True,pad_inches = 0,dpi = 300,bbox_inches = 'tight')
    df = pd.read_csv("result2.csv",sep='\t',encoding="utf-8")
    df['class_name'] = df['class_name'].astype(str)
    df['new'] = df['location']+'\t'+df['class_name']
    df = df.groupby(by='image_name').apply(lambda x:('\t'.join(x['new'])))
    df.to_csv("test.csv",sep='\t')
    file1 = open('test.csv','r').readlines()
    fileout = open('result2.csv','w')
    for line in file1:
    	fileout.write(line.replace('"',''))
    fileout.close()
    #plt.show()
