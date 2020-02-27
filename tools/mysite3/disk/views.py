# -*- coding:utf-8 -*-
#Author: Wu Jiang

from django.views.decorators.csrf import csrf_exempt
import _init_paths       #导入路径，调用faster_rcnn中相关模块
from django.shortcuts import render,render_to_response
from django import forms         #Django表单方式上传图片
from django.http import HttpResponse
from disk.models import User                        
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
#from utils.timer import Timer  by wj 11.3
import matplotlib.pyplot as plt   #matlab中画图工具
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys
import glob
#from skimage import io,data   #11.19 by wj
#import shutil      #11.27 by wj  file
from os import listdir  #11.28 by wj    上述导入模块有三个来源：一是views.py源文件、二是从demo中拷贝过来、三是为了后面读写文件操作

# Create your views here.
class UserForm(forms.Form):
    username = forms.CharField()
    headImg = forms.FileField()    
    
#直接从demo中拷贝来的
CLASSES = ('__background__',
           '1','2','3','4','5','6','7','8','9','10','11','14','15','16','17','18','19','20','others')

NETS = {'ResNet-50': ('ResNet-50',
                  'ResNet-50_faster_rcnn_iter_110000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
                  
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))     
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),         
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file ='./upload/'+image_name  #处理图片过程中,传参
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    #timer = Timer()  by wj 11.3
    #timer.tic()   by wj 11.3
    scores, boxes = im_detect(net, im)
    #timer.toc()   by wj 11.3
    #print ('Detection took {:.3f}s for '  by wj 11.3
           #'{:d} object proposals').format(timer.total_time, boxes.shape[0])   by wj 11.3

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.savefig('/data/yinzg/faster/tools/mysite3/static/test.jpg')   #图片处理后保存下来
    plt.savefig("/data/yinzg/faster/tools/mysite3/static/" + image_name, format = 'jpg',transparent = True,pad_inches = 0,dpi = 300,bbox_inches = 'tight')

def parse_args_test():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-50]',
                        choices=NETS.keys(), default='ResNet-50')

    args = parser.parse_args()

    return args

@csrf_exempt
def register(request):
    if request.method == "POST":
        uf = UserForm(request.POST,request.FILES)
        #print sys.path
        if uf.is_valid():
            #获取表单信息
            username = uf.cleaned_data['username']
            headImg = uf.cleaned_data['headImg']     #stand for dir or photo ??   request.FILES['file'] acquire photo  ?
            #写入数据库
            user = User()
            user.username = username
            user.headImg = headImg
            user.save()
			
            #a=listdir('./upload')    #11.28 by wj                   将传输过来的图片改名成000001.jpg
            #old_file_path=os.path.join('./upload',a[0]) #11.28 by wj将传输过来的图片改名成000001.jpg
            #new_file_path='./upload/000001.jpg'  #11.28 by wj       将传输过来的图片改名成000001.jpg
            #os.rename(old_file_path, new_file_path)  #11.28 by wj   将传输过来的图片改名成000001.jpg
			

            #if __name__ == '__main__':
            cfg.TEST.HAS_RPN = True  # Use RPN for proposals

            #args = parse_args_test()
            #args=Namespace(cpu_mode=False, demo_net='zf', gpu_id=0)

            #prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                           # 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
            prototxt ='/data/yinzg/faster/models/pascal_voc/ResNet-50/faster_rcnn_end2end/test.prototxt'
            #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              #NETS[args.demo_net][1])
            caffemodel ='/data/yinzg/faster/data/faster_rcnn_models/ResNet-50_faster_rcnn_iter_110000.caffemodel'

    #if not os.path.isfile(caffemodel):
        #raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       #'fetch_faster_rcnn_models.sh?').format(caffemodel))   by wj 17.11.8

    #if args.cpu_mode:
            #caffe.set_mode_cpu()
    #else:
            caffe.set_mode_gpu()
            caffe.set_device(1)
            cfg.GPU_ID = 1
            net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    #print '\n\nLoaded network {:s}'.format(caffemodel)  by wj 17.11.8

    # Warmup on a dummy image
            im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
            for i in xrange(2):
                 _, _= im_detect(net, im)

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',  by wj 11.3
               # '001763.jpg', '004545.jpg']   boy wj 11.3
            im_names = os.listdir("./upload") #处理此文件夹的图片
            #im_names = ['000001.jpg']     #固定处理名为000001.jpg的图片
            for im_name in im_names:
        #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'  by wj 17.11.8
        #print 'Demo for data/demo/{}'.format(im_name) by wj 17.11.8
        #print args   how to change this
                demo(net, im_name)
                imagepath ='/data/yinzg/faster/tools/mysite3/static/'+im_name
				
            #plt.show()
            #os.remove(new_file_path)   #11.28 by wj    在上传新的图片前清空upload文件夹
            os.remove('/data/yinzg/faster/tools/mysite3/upload/'+im_name)
            #return HttpResponse('upload ok!') #11.29 by wj 
            #imagepath ='/data/yinzg/faster/tools/mysite3/static'
            image_data = open(imagepath,"rb").read()  
            return HttpResponse(image_data,content_type="image/jpg") 
    else:
         uf = UserForm()
    return render_to_response('register.html',{'uf':uf})