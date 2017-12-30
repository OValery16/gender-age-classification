from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six.moves
from datetime import datetime
import sys
import math
import time
import timeit
#from data import inputs, standardize_image
import numpy as np
import tensorflow as tf
import re
import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2
from numpy import array
from distutils.version import LooseVersion
import os

VERSION_GTE_0_12_0 = LooseVersion(tf.__version__) >= LooseVersion('0.12.0')

# Name change in TF v 0.12.0
if VERSION_GTE_0_12_0:
    standardize_image = tf.image.per_image_standardization
else:
    standardize_image = tf.image.per_image_whitening

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def normalize(image):
    image = image / 255.
    
    return image

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def draw_reduced(image, boxes, labels,name):
    
    i=0
    for box in boxes:
        i=i+1
        maxDist=max((box.w*image.shape[1]),(box.h)* image.shape[0])
        xmin  = int((box.x * image.shape[1])- maxDist/2) 
        xmax  = int((box.x * image.shape[1])+ maxDist/2) 
        ymin  = int((box.y * image.shape[0])- maxDist/2) 
        ymax  = int((box.y * image.shape[0])+ maxDist/2)
        
        cv2.imwrite(name, image[ymin:ymax,xmin:xmax])
    return image    

def draw_boxes(image, boxes, labels):
    
    i=0
    for box in boxes:
        i=i+1
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        
        #cv2.putText(image, 
        #            labels[box.get_label()] + ' ' + str(box.get_score()), 
        #            (xmin, ymin - 13), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 
        #            1e-3 * image.shape[0], 
        #            (0,255,0), 2)
        print(labels[box.get_label()] + ' ' +str(box.get_score()) + ' size: '+str(xmax-xmin)+'X'+str(ymax-ymin) )
        cv2.imwrite('/home/olivier/Desktop/yolo-floydhub/test/image_extracted'+str(i)+'.jpg', image[ymin:ymax,xmin:xmax])
        #temp =image[int((box.y* image.shape[1]-64)):int((box.y* image.shape[1]+64)),int((box.x* image.shape[1]-64)):int((box.x* image.shape[1]+64))]
        #cv2.imwrite('/home/olivier/Desktop/yolo-floydhub/test/image_extracted'+str(i)+'.jpg', temp)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        #cv2.rectangle(image, (int((box.x* image.shape[1]-64)),int((box.y* image.shape[1]-64))), (int((box.x* image.shape[1]+64)),int((box.y* image.shape[1]+64))), (0,255,0), 3)
    return image        

def getFacesList(image, boxes):
    imgList=[]
    i=0
    for box in boxes:
        i=i+1
        if (len(boxes) ==1):
            maxDist=max((box.w*image.shape[1]),(box.h)* image.shape[0])*1.4
        else:
            maxDist=max((box.w*image.shape[1]),(box.h)* image.shape[0])*1.05
        xmin  = int((box.x * image.shape[1])- maxDist/2) 
        xmax  = int((box.x * image.shape[1])+ maxDist/2) 
        ymin  = int((box.y * image.shape[0])- maxDist/2) 
        ymax  = int((box.y * image.shape[0])+ maxDist/2)
        
        height, width = image.shape[:2]
        #adjustment
        if (xmin<0):
            print("xmin<0")
            dif=xmin
            xmin=max(xmin,0)
            ymin=int(round(ymin-dif/2))
            ymax=int(round(ymax+dif/2))
        if (ymin<0):
            print("ymin<0",)
            dif=ymin
            ymin=max(ymin,0)
            xmin=int(round(xmin-dif/2))
            xmax=int(round(xmax+dif/2)) 
            print("ymin<0 xmin="+str(xmin)+"xmax"+str(xmax)+"dist1 ="+str(xmax-xmin)+"dist2="+str(ymax-ymin))
        if (xmax>width):
            print("xmax>width")
            dif=width-xmax
            xmax=min(xmax,width )
            ymin=int(round(ymin-dif/2))
            ymax=int(round(ymax+dif/2))
        if (ymax>height):
            print("xmax>height")
            dif=width-ymax
            ymin=min(ymax,height )
            xmin=int(round(xmin-dif/2))
            xmax=int(round(xmax+dif/2 ))            
        
        
        #xmax=min(xmax,width )
        #ymin=max(ymin,0)
        #ymax=min(ymax,height )
        
        #print("width: "+str(width)+" height: "+str(height)+" xmin: "+str(xmin)+ " xmax: "+str(xmax)+" ymin: "+str(ymin)+ " ymax: "+str(ymax) )
        
        path = os.path.dirname(os.path.abspath(__file__))+'/detected_faces/image_extracted'+str(i)+'.jpg'
        cv2.imwrite(path, image[ymin:ymax,xmin:xmax])
        imgList.append(path)

    return imgList        

def draw_boxes_v2(image, boxes, labels,listPrediction):
    
    i=0
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        temp=array( listPrediction[i][3])
        if temp[0]>temp[1]:
           gender="Female"
        else:
           gender="Male"
        
        (gender +' ' + str(listPrediction[i][1]) +' '+ str(listPrediction[i][0]))
              
        a=int((xmin+xmax)/2)-10
        cv2.putText(image,gender +' ' + str(listPrediction[i][1]),
        (xmin, ymax+25),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=1e-3 *3* image.shape[0], 
        color=(0, 255, 0))          
        i=i+1
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #del listPrediction
        del gender
        del box

        
    return image 
        
def decode_netout(netout, obj_threshold, nms_threshold, anchors, nb_class):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x, y, w, h, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in xrange(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in xrange(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes

def decode_netout2( netout,labels,anchors, obj_threshold=0.4, nms_threshold=0.3):
        grid_h, grid_w, nb_box = netout.shape[:3]
        nb_class=len(labels)
        boxes = []
        
        # decode the output by the network
        netout[..., 4]  = sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_threshold
        
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                        w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                        confidence = netout[row,col,b,4]
                        
                        box = BoundBox(x, y, w, h, confidence, classes)
                        
                        boxes.append(box)

        # suppress non-maximal boxes
        for c in range(nb_class):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

            for i in xrange(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in xrange(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                            boxes[index_j].classes[c] = 0
                            
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > obj_threshold]
        
        return boxes

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)



RESIZE_AOI = 256
RESIZE_FINAL = 227

# Modifed from here
# http://stackoverflow.com/questions/3160699/python-progress-bar#3160819
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='='):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def update(self, step=1):
        self.current += step
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        six.print_('\r' + self.fmt % args, end='')

    def done(self):
        self.current = self.total
        self.update(step=0)
        print('')

# Read image files            
class ImageCoder(object):
    
    def __init__(self):
        # Create a single Session to run all image coding calls.
        config = tf.ConfigProto(allow_soft_placement=True)
        self._sess = tf.Session(config=config)
        
        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
        
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self.crop = tf.image.resize_images(self._decode_jpeg, (RESIZE_AOI, RESIZE_AOI))

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self.crop, #self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})

        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
        

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename
        
def make_multi_image_batch(filenames, coder):
    """Process a multi-image batch, each with a single-look
    Args:
    filenames: list of paths
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """

    images = []

    for filename in filenames:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            image_data = f.read()
        # Convert any PNG to JPEG's for consistency.
        if _is_png(filename):
            print('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)
    
        image = coder.decode_jpeg(image_data)

        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        image = standardize_image(crop)
        images.append(image)
    image_batch = tf.stack(images)
    return image_batch

def make_multi_crop_batch(filename, coder):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """
    
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
    
    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    
    image = coder.decode_jpeg(image_data)
    crops = []
    print('Running multi-cropped image')
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(tf.image.flip_left_right(crop))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = tf.image.flip_left_right(cropped)
        crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)
    return image_batch

