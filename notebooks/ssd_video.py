# coding: utf-8
import os
import math
import random
 
import numpy as np
import tensorflow as tf
import cv2
 
slim = tf.contrib.slim
 
import sys
sys.path.append('../')
 
from nets import ssd_vgg_300, ssd_common, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
# draw boxes
from notebooks import visualization_camera
 
 
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
 
# Input placeholder. use 300*300
#net_shape = (300, 300)
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)
 
# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
#ssd_net = ssd_vgg_300.SSDNet()
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/model/model.ckpt-314913'
#ckpt_filename = '../checkpoints/model11/model.ckpt-75822'
#ckpt_filename = '../checkpoints/model9/model.ckpt-7171'
#ckpt_filename = '../checkpoints/model7/model.ckpt-92084'
# ckpt_filename = '../checkpoints/model6/model.ckpt-72392'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
#NMS——非极大值抑制
def process_image(img, select_threshold=0.8, nms_threshold=.2, net_shape=(512, 512)):
#def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
 
# following are added for camera demo
cap = cv2.VideoCapture(r'1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
# number_of_frames = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
print('fps=%d,size=%r,fourcc=%r' % (fps, size, fourcc))
#delay = int(30/int(fps))
delay=25
i = 1
# picture is too large
width = int(size[0])
height = int(size[1])

#width = 300
#height = 300
 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True and i > 3296:
        image_np = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        # Actual detection.
        rclasses, rscores, rbboxes =  process_image(image_np)
        # Visualization of the results of a detection.
        visualization_camera.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
        pic_name = "D:/DeepLearning/SSD-Tensorflow/picture_process/" + str(i) + ".jpg"
        #cv2.imshow('Detecting Test of Vehicle', image_np)
        cv2.imwrite(pic_name, image_np)
        # delay delay ms
        #cv2.waitKey(delay)
        print('Ongoing...', i)
    i = i + 1


cap.release()
cv2.destroyAllWindows()
