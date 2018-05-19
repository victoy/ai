import os
import tarfile
import requests

'''
Tensorflow provide this model zoo. 
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

This is the script to download all the models. 
'''
# What model to download.
MODEL_NAME = ['ssd_mobilenet_v1_coco_2017_11_17'
                ,'ssd_mobilenet_v2_coco_2018_03_29'
                ,'ssdlite_mobilenet_v2_coco_2018_05_09'
                ,'ssd_inception_v2_coco_2017_11_17'
                ,'faster_rcnn_inception_v2_coco_2018_01_28'
                ,'faster_rcnn_resnet50_coco_2018_01_28'
                ,'faster_rcnn_resnet50_lowproposals_coco_2018_01_28'
                ,'rfcn_resnet101_coco_2018_01_28'
                ,'faster_rcnn_resnet101_coco_2018_01_28'
                ,'faster_rcnn_resnet101_lowproposals_coco_2018_01_28'
                ,'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
                ,'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28'
                ,'faster_rcnn_nas_coco_2018_01_28'
                ,'faster_rcnn_nas_lowproposals_coco_2018_01_28'
                ,'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
                ,'mask_rcnn_inception_v2_coco_2018_01_28'
                ,'mask_rcnn_resnet101_atrous_coco_2018_01_28'
                ,'mask_rcnn_resnet50_atrous_coco_2018_01_28'
                ,'faster_rcnn_resnet101_kitti_2018_01_28'
                ,'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
                ,'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28'
                ,'faster_rcnn_resnet101_ava_v2.1_2018_04_30']
MODEL_FILE_EXT = '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

def download(url, path, dest):
    '''
    Extract only 'pb' file from the downloaded tarfile.
    :param url: google model zoo url
    :param path: local destination full path
    :param dest: destination folder
    :return:
    '''
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)
    tar_file = tarfile.open(path)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, dest)


for i in range(len(MODEL_NAME)):
    MODEL_FILE = MODEL_NAME[i] + MODEL_FILE_EXT
    download(DOWNLOAD_BASE + MODEL_FILE, os.getcwd() + '/' + MODEL_FILE, os.getcwd())
    print('[Done] '+MODEL_FILE)