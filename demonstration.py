import tensorflow
import os
import sys
import random
import math
import numpy as np
import argparse
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
# Submodule Libraries
CITI_ROOT = os.path.abspath('cityscapesScripts/')
MASK_ROOT = os.path.abspath('Mask_RCNN/')

sys.path.append(MASK_ROOT)
sys.path.append(CITI_ROOT)

# Import Mask RCNN
sys.path.append(MASK_ROOT)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(MASK_ROOT, 'samples/coco/'))  # To find local version

#Import Submodule Libraries

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import coco
from CityScapesDataset import CityscapesSegmentationDataset, TrainingConfig

import cv2

# Globals
class_names = ['ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
               'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
               'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

# Directory to save logs and trained model
MODEL_DIR = os.path.join('logs')

# Root directory of the project
ROOT_DIR = os.path.abspath('Mask_RCNN/')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def load_model(model_path):
    config = TrainingConfig()
    config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = 1

    model = modellib.MaskRCNN(mode='inference', config=config, model_dir=ROOT_DIR)
    #model.load_weights(COCO_MODEL_PATH, by_name=True)
    model.load_weights(model_path, by_name=True)

    return model

def save_frame(vw, frame):

    pass

def draw_instances(image, boxes, masks, classes, scores, colors):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == classes.shape[0]

    masked_image = image #image.astype(np.uint32).copy()
    
    for i in range(N):
        color = colors[classes[i]]
        if scores[i] < 0.9:
            continue
        # Mask
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color)
    
    return masked_image

def video_loop(vc, vw, model, display_video, record):
    rval = True
    colors = visualize.random_colors(35)
    if display_video:
        cv2.namedWindow('instances_demo', cv2.WINDOW_NORMAL)
        #cv2.namedWindow('video_demo', cv2.WINDOW_NORMAL)
    count = 0
    while rval:
        
        if count % 30 != 0:
            rval, frame = vc.read()
            if cv2.waitKey(25) & 0xFF == ord('q'):
	            break
            count+=1
            continue
        rval, frame = vc.read()


        results = model.detect([frame])[0]
        masked_matrix = draw_instances(frame, results['rois'], results['masks'], results['class_ids'], results['scores'], colors)
        
        print(masked_matrix.shape)
        if display_video:
            cv2.imshow('instances_demo', masked_matrix)
            #cv2.imshow('video_demo', frame)
        if record:
            vw.write(masked_matrix)
            print('saving frame')
        if cv2.waitKey(25) & 0xFF == ord('q'):
	        break
        count+=1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_model', required=True ,action='store')
    parser.add_argument('--recordOnly', action='store_false')
    parser.add_argument('--record', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    vc = cv2.VideoCapture('/home/jeff/Downloads/20200425_181735.mp4')
    vw = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 2.0, (1920,1080))
    model = load_model(args.path_to_model)

    video_loop(vc, vw, model, args.recordOnly, args.record)

    vc.release()
    vw.release()
if __name__ == "__main__":
    main()
