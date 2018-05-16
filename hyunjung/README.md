# Object Detection

[Modern History of Object Recognition](https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318)

There are multiple object detection frameworks like below. This project includes some examples to use the framworks.
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)  
* [Detectron](https://github.com/facebookresearch/Detectron)
* [Yolo Darknet](https://github.com/pjreddie/darknet)
* [opencv](https://github.com/opencv/opencv)

### Tensorflow Object Detection API
I assume we already have tensorflow. If not, you could reference this [installation guide](https://www.tensorflow.org/install/install_mac). 
1. Clone [tensorflow models](https://github.com/tensorflow/models) ( Use this directory for #2)
2. [Install Tensorflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
 
 cf) You might like to add 'tensorflow/models/research' path to PYTHONPATH. If you add this into bash file and your editor, it would be helpful for your future work. 
 [add to PYTHONPATH in PyCharm](https://stackoverflow.com/questions/17198319/how-to-configure-custom-pythonpath-with-vm-and-pycharm)
 
 
## Networks

| Model name  | Architecture|
| ------------ | :--------------: |
| [AlextNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | ![AlexNet](https://kratzert.github.io/images/finetune_alexnet/alexnet.png)|
| [Vgg 16](https://arxiv.org/abs/1409.1556) | ![VGG16](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)|
| [SSD](https://arxiv.org/abs/1512.02325) | ![SSD](https://i.stack.imgur.com/1IBy4.png)|
| [ResNet](http://arxiv.org/abs/1512.03385)|![Resnet](https://image.slidesharecdn.com/lenettoresnet-170509055515/95/lenet-to-resnet-17-638.jpg)|
| [MobileNet](https://arxiv.org/abs/1704.04861)|![MobileNet](http://machinethink.net/images/mobilenets/Architecture@2x.png) |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | ![fasterrcnn](https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/faster-rcnn.png)|
| [Mask RNN](https://arxiv.org/pdf/1703.06870.pdf) | ![MaskRNN](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/56b614839510bbf672a6ce3dc6bafbf0d1bc7629/4-Figure2-1.png)|