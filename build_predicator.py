from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import timeit
#from data import inputs
import numpy as np
import tensorflow as tf
from model import get_checkpoint, inception_v3,inception_v3_test
from utils import *
import os
import json
import csv
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
#from preprocessing import parse_annotation 
from utils import draw_boxes, draw_boxes_v2,getFacesList
from modelGender import genderClassifier
from build_predicator import *
import json
import time
import matplotlib.pyplot as plt

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

MODEL_DIR="ageWeights"
genderpath='genderWeights/testsave.meta'
genderckp='genderWeights/'
CLASS_TYPE='age'
#FILENAME="/home/olivier/Desktop/age_estimation2/test2/image/olivier.jpg"
TARGET=''
DEVICE_ID='/gpu:0'
CHECKPOINT='checkpoint'
REQUESTED_STEP=''
SINGLE_LOOK=True
FACE_DETECTION_MODEL=''
FACE_DETECTION_TYPE='cascade'

class Graph(object):

    def classify_age(self,image_files):
        label_list=self.label_list
        softmax_output=self.softmax_output
        coder=self.coder
        images=self.images
        feature=self.feature
        writer=self.writer
        listPrediction=[]
        listPrediction=[]
        try:
            num_batches = int(math.ceil(len(image_files) / 128))
            #pg = ProgressBar(num_batches)
            for j in range(num_batches):
                start_offset = j * 128
                end_offset = min((j + 1) * 128, len(image_files))
                batch_image_files = image_files[start_offset:end_offset]
                image_batch = make_multi_image_batch(batch_image_files, coder)
                batch_results,featureData = self._sess.run([softmax_output,feature], feed_dict={images:image_batch.eval(session=self._sess_default)})
                batch_predictionG = self._sessGender.run(self.prediction, feed_dict={self.input: featureData})
                batch_sz = batch_results.shape[0]
                #if several image => iterate
                for i in range(batch_sz):
                    output_i = batch_results[i]
                    best_i = np.argmax(output_i)
                    best_choice = (label_list[best_i], output_i[best_i])
                    #print('Guess @ 1 %s, prob = %.2f' % best_choice)
                    listPrediction.append((i,label_list[best_i], output_i[best_i],batch_predictionG[i]))
                    if writer is not None:
                        f = batch_image_files[i]
                        writer.writerow((f, best_choice[0], '%.2f' % best_choice[1]))
            del batch_predictionG
            del batch_results
            del featureData
            del batch_image_files
            
        except Exception as e:
            print(e)
            print('Failed to run all images')

        return listPrediction

    def __init__(self):
        #don t forget to reset graph or error (olivier)
        #tf.reset_default_graph()
        self._sess_default=tf.Session()
        self.g_1 = tf.Graph()
        with self.g_1.as_default():

            files = []
            config = tf.ConfigProto(allow_soft_placement=True)
            self._sess= tf.Session(config=config)

            label_list = AGE_LIST
            nlabels = len(label_list)

            print('Executing on %s' % DEVICE_ID)
            with tf.device(DEVICE_ID):

                images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                logits, feature = inception_v3_test(nlabels, images, 1, False)
                init = tf.global_variables_initializer()

                 #restore a specific "checkpoint" (which step) olivier
                requested_step = None

                checkpoint_path = MODEL_DIR

                model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, CHECKPOINT)
                #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
                saver = tf.train.Saver()
                saver.restore(self._sess, model_checkpoint_path)

                softmax_output = tf.nn.softmax(logits)
                coder = ImageCoder()

                #image_files=[FILENAME,FILENAME,FILENAME]

                writer = None

                #check tensorboard (olivier)
                writerTB = tf.summary.FileWriter('log', graph=tf.get_default_graph())
                self.label_list= label_list
                self.softmax_output= softmax_output
                self.coder= coder
                self.images=images, 
                self.writer=writer
                self.feature=feature
                
       
        self.g_2 = tf.Graph()
        with self.g_2.as_default():
            self._sessGender= tf.Session()
            self.prediction,self.input=genderClassifier()
            init = tf.global_variables_initializer()
            new_saver = tf.train.import_meta_graph(genderpath)
            new_saver.restore(self._sessGender, tf.train.latest_checkpoint(genderckp))
       

    
    def close_sess(self): 
        #sess.close()
        self._sess_default.close()
        self._sessGender.close()



