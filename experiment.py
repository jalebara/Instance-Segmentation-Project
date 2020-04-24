import tensorflow
import os
import sys
import random
import math
import numpy as np
import imgaug
import argparse
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from types import MethodType

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
from mrcnn.config import Config

import coco
from CityScapesDataset import CityscapesSegmentationDataset, TrainingConfig, EvaluationConfig

#Global Constants
# Root directory of the project
ROOT_DIR = os.path.abspath('Mask_RCNN/')



# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

class_names = ['ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
               'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
               'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

class Experiment():
    def __init__(
                self,
                name, 
                results_path,
                image_size_min,
                image_size_max,
                images_per_gpu, 
                learning_rate,
                epochs,
                layers_to_train,
                augmentation,
                root_data_directory,
                training_func = None):

        class ExperimentConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = images_per_gpu
            LEARNING_RATE = learning_rate
            NUM_CLASSES = 35 #1+34 # Background (inherited from utils.Dataset) + FG classes (listed below)
            IMAGE_MIN_DIM = image_size_min
            IMAGE_MAX_DIM = image_size_max
            NAME = 'cityscape'
        class TestingConfig(Config):
            # TO-OPT: Set batch size to 20 by default.
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NAME = 'cityscape'
            NUM_CLASSES = 35
            IMAGE_MIN_DIM = image_size_min
            IMAGE_MAX_DIM = image_size_max

        self.experiment_config = ExperimentConfig()
        self.results_path = results_path
        self.name = name
        self.epochs = epochs
        self.layers = layers_to_train
        self.augmentation = augmentation
        self.data_dir = root_data_directory
        self.model_path = COCO_MODEL_PATH
        self.model_save_dir = os.path.join(self.results_path, 'logs')
        self.testing_config = TestingConfig()
        self.history = None
        if training_func is not None:
            self.training_func = MethodType(training_func, self)
    
    def _get_latest_checkpoint(self):
        dir_names = next(os.walk(self.model_save_dir))[1]
        key = self.experiment_config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        print(dir_names)

        if not dir_names:
            return COCO_MODEL_PATH # no weights trained
        fps = []
        # Pick last directory
        for d in dir_names: 
            dir_name = os.path.join(self.model_save_dir, d)
            # Find the last checkpoint
            checkpoints = next(os.walk(dir_name))[2]
            checkpoints = filter(lambda f: f.endswith('.h5'), checkpoints)
            checkpoints = list(reversed(sorted(checkpoints)))
            if not checkpoints:
                print('No weight files in {}'.format(dir_name))
            else:
                checkpoint = os.path.join(dir_name, checkpoints[0])
                fps.append(checkpoint)
        if fps is None:
            #empty list
            return COCO_MODEL_PATH
            
        model_path = sorted(fps)[0]
        print('Found models {}'.format(str(fps)))
        return model_path
    
    def prepare(self):
        if os.path.isdir(self.results_path):
            self.model_path = self._get_latest_checkpoint() # replace default MS COCO weights with latest weights
        
        else:
            #Experiment has not started, so we should make the directories
            os.makedirs(self.results_path)
            os.makedirs(self.model_save_dir)
    
    def run(self):
        print('Running Training')
        # check if experiment has completed
        if self.model_path != COCO_MODEL_PATH:
            epoch_num = 1 + int(self.model_path[-7:-3])
            print('checkpoint epoch is {}'.format(int(self.model_path[-7:-3])))
            if epoch_num == self.epochs:
                print('Training completed')
                return

        # Create model object in training mode.
        model = modellib.MaskRCNN(mode="training", model_dir=self.model_save_dir, config=self.experiment_config)
        
        if self.model_path == COCO_MODEL_PATH:
            # Load weights trained on MS-COCO, excepting areas for training
            # We can exclude the bounding box layers for now, but they will
            # be useful for interpreting our images for now
            model.load_weights(self.model_path, by_name=True, exclude=["mrcnn_bbox_fc",
                                                                    "mrcnn_bbox",
                                                                    "mrcnn_mask",
                                                                    "mrcnn_class_logits"])
        else:
            model.load_weights(self.model_path, by_name=True)           


        dataset_train = CityscapesSegmentationDataset()
        dataset_train.load_cityscapes(self.data_dir, 'train')
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CityscapesSegmentationDataset()
        dataset_val.load_cityscapes(self.data_dir, 'val')
        dataset_val.prepare()

        model = self.training_func(model, dataset_train, dataset_val)

        # Retrieve history for plotting loss and accuracy per epoch
        self.history = model.keras_model.history.history

    def save_results(self):
        if os.path.exists(os.path.join(self.results_path, "{}-experiment-loss-curve.png".format(self.name))):
            return #already saved results
        print("Computing and saving results")
        if self.history is not None:
            with open(os.path.join(self.results_path, '{}-training_history'.format(datetime.now().strftime('%Y-%m-%d-%H-%M'))), 'wb') as file_pi:
                pickle.dump(self.history, file_pi)
            
            #Generate Loss Curves
            plt.figure('loss_curve')
            plt.plot(self.history['loss'])
            plt.plot(self.history['val_loss'])
            plt.title('Training and Validation Loss vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(('Training', 'Validation'))
            plt.savefig(os.path.join(self.results_path, "{}-experiment-loss-curve.png".format(self.name)))
        
        model_checkpoint_path = self._get_latest_checkpoint()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode='inference', 
                                config=self.testing_config,
                                model_dir=MASK_ROOT)
        model.load_weights(model_checkpoint_path, by_name=True)

        #Testing dataset.
        dataset_test = CityscapesSegmentationDataset()
        dataset_test.load_cityscapes(self.data_dir, 'test')
        dataset_test.prepare()

        # Compute VOC-Style mAP @ IoU=0.5
        # Running on 10 images. Increase for better accuracy.
        #image_ids = np.random.choice(dataset_test.image_ids, 20)
        APs = []
        for image_id in dataset_test.image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, self.testing_config,
                                    image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, self.testing_config), 0)
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
        mAP = np.mean(APs)        
        print("mAP: {}".format(mAP))
        
        
        with open(os.path.join(self.results_path, "{0}-experiment-config-{1}.txt".format(self.name, datetime.now().strftime('%Y-%m-%d-%H-%M'))),'w') as f:
            f.write("Learning Rate: {} \n".format(self.experiment_config.LEARNING_RATE))
            f.write("Epochs: {} \n".format(self.epochs))
            f.write("Layers: {} \n".format(str(self.layers)))
            if self.augmentation is not None:
                f.write("Augmentation: See Experiment file\n")
            else:
                f.write("Augmentation: None\n")
            f.write("Max Image Size: {}\n".format(self.experiment_config.IMAGE_MAX_DIM))
            f.write("Min Image Size: {}\n".format(self.experiment_config.IMAGE_MIN_DIM))
            f.write("Images per GPU: {}\n".format(self.experiment_config.IMAGES_PER_GPU))
            f.write("RESULTS\n")
            f.write("Mean AP: {}\n".format(mAP))
    
    def training_func(self, model, dataset_train, dataset_val):
        model.train(dataset_train, 
            dataset_val,
            learning_rate=0.005,
            epochs=20,
            layers='heads',
            augmentation=self.augmentation)
        model.train(dataset_train, 
            dataset_val,
            learning_rate=0.001,
            epochs=25,
            layers='4+',
            augmentation=self.augmentation)
        return model

if __name__ == "__main__":
    experiment1_config = {
        "name": 'high-lr-no-augment', 
        "results_path": '/home/jabaraho/coding/ECE542FinalProject/logs/experiment1',
        "image_size_min": 512,
        "image_size_max": 512,
        "images_per_gpu": 2, 
        "learning_rate": 0.005,
        "epochs": 2,
        "layers_to_train": 'heads',
        "augmentation": None,
        "root_data_directory": '/home/jabaraho/coding/ECE542FinalProject/data'
    }

    experiment2_config = {
        "name": 'high-lr-augment', 
        "results_path": '/home/jabaraho/coding/ECE542FinalProject/logs/experiment2',
        "image_size_min": 512,
        "image_size_max": 512,
        "images_per_gpu": 2, 
        "learning_rate": 0.005,
        "epochs": 2,
        "layers_to_train": 'heads',
        "augmentation": imgaug.augmenters.Sometimes(0.5, [
                            imgaug.augmenters.Fliplr(0.5),
                            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                        ]),
        "root_data_directory": '/home/jabaraho/coding/ECE542FinalProject/data'
    }

    experiment3_config = {
        "name": 'low-lr-no-augment', 
        "results_path": '/home/jabaraho/coding/ECE542FinalProject/logs/experiment3',
        "image_size_min": 512,
        "image_size_max": 512,
        "images_per_gpu": 2, 
        "learning_rate": 0.001,
        "epochs": 2,
        "layers_to_train": '4+',
        "augmentation": None,
        "root_data_directory": '/home/jabaraho/coding/ECE542FinalProject/data'
    }

    experiment4_config = {
        "name": 'low-lr-augment', 
        "results_path": '/home/jabaraho/coding/ECE542FinalProject/logs/experiment2',
        "image_size_min": 512,
        "image_size_max": 512,
        "images_per_gpu": 2, 
        "learning_rate": 0.001,
        "epochs": 2,
        "layers_to_train": '4+',
        "augmentation": imgaug.augmenters.Sometimes(0.5, [
                            imgaug.augmenters.Fliplr(0.5),
                            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                        ]),
        "root_data_directory": '/home/jabaraho/coding/ECE542FinalProject/data'
    }

    experiment5_config = {
        "name": 'low-lr-augment', 
        "results_path": '/home/jabaraho/coding/ECE542FinalProject/logs/experiment2',
        "image_size_min": 512,
        "image_size_max": 512,
        "images_per_gpu": 2, 
        "learning_rate": 0.001,
        "epochs": 2,
        "layers_to_train": '4+',
        "augmentation": imgaug.augmenters.Sometimes(0.5, [
                            imgaug.augmenters.Fliplr(0.5),
                            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                        ]),
        "root_data_directory": '/home/jabaraho/coding/ECE542FinalProject/data'
    }

    experiment_configs = [experiment5_config]

    for ex_conf in experiment_configs:
        experiment = Experiment(**ex_conf)
        experiment.prepare()
        experiment.run()
        experiment.save_results()