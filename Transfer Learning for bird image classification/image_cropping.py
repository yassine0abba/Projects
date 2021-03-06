# -*- coding: utf-8 -*-
"""image_Cropping.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Aj2Shn4Hj4VdqKe5iR9QVwepQp3RxLo2
"""

from google.colab import drive
drive.mount('/content/drive/')



#!pip install chainercv
from chainercv.datasets import voc_bbox_label_names
import numpy as np
from chainercv.links import FasterRCNNVGG16, SSD300, SSD512, YOLOv3, YOLOv2
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox
from PIL import Image
import os

"""# Cropping models

"""

models = [YOLOv2(pretrained_model='voc0712'),YOLOv3(pretrained_model='voc0712'),YOLOv3(pretrained_model='voc0712'),SSD300(pretrained_model='voc0712'),SSD512(pretrained_model='voc0712')]

root='/content/drive/My Drive/MVA/Object recognition/bird_dataset/train_images/'
dirlist = np.sort([ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ])

def test_model(model, img):
    crop, detected_labl, prob = model.predict([img])
    if len(detected_labl[0])>0 and detected_labl[0][0]==2:  # Label= 2 = bird
          return prob[0][0], crop
    return(-1,-1)

Image.open('/content/drive/MyDrive/MVA/Object recognition/bird_dataset/train_images/004.Groove_billed_Ani/Groove_Billed_Ani_0068_1538.jpg')

for k in range(len(dirlist)):
  folder = root + dirlist[k]+'/'
  for im  in os.listdir(folder):
    img = read_image(folder+im)
    output = [test_model(model,img) for model in models]
    prob = [x[0] for x in output]
    max_prob = np.argmax(prob)
    crop = output[max_prob][1][0][0]
    img_ = Image.open(folder+im)
    cropped = img_.crop( ( crop[0], crop[1], crop[2], crop[3]) ) 
    try :
      cropped.save(folder+im, "JPEG", quality=80, optimize=True, progressive=True)
    except :
      PIL.ImageFile.MAXBLOCK = cropped.size[0] * cropped.size[1]
      cropped.save(folder+im, "JPEG", quality=80, optimize=True, progressive=True)

