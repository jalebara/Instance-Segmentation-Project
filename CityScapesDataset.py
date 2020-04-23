
from mrcnn.config import Config
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import os
import random

from tqdm import tqdm    
from mrcnn.utils import Dataset
from cityscapesscripts.helpers.csHelpers import getCoreImageFileName

GPU_COUNT = 1
IMAGES_PER_GPU = 2
LEARNING_RATE = 0.0001
NAME = 'cityscape'
NUM_CLASSES = 35 #1+34 # Background (inherited from utils.Dataset) + FG classes (listed below)
WEIGHT_DECAY = 0.0001

class TrainingConfig(Config):
    # TO-OPT: Set batch size to 20 by default.
    GPU_COUNT = GPU_COUNT
    IMAGES_PER_GPU = IMAGES_PER_GPU
    LEARNING_RATE = LEARNING_RATE
    NAME = NAME
    NUM_CLASSES = NUM_CLASSES
    WEIGHT_DECAY = WEIGHT_DECAY
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

config = TrainingConfig()
config.display()

class_names = ['ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
               'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
               'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

class CityscapesSegmentationDataset(Dataset):
    
    def load_cityscapes(self, root_directory, subset):

        # add class names
        class_names = ['ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
               'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
               'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']
        
        for i, name in enumerate(class_names[:-1]):
            self.add_class('cityscape', i, name)
        
        # license plate has id as -1
        self.add_class('cityscape',-1,class_names[-1])

        # Write out locations for annotations and images
        # self.data_dir: location for json annotations
        # image_dir: location for image path assignment
        if subset == 'train':
            self.data_dir = os.path.join(root_directory, 'train')
            image_dir = os.path.join(self.data_dir, 'train')
        elif subset == 'val':
            self.data_dir = os.path.join(root_directory, 'val')
            image_dir = os.path.join(self.data_dir, 'val')
        elif subset == 'test':
            self.data_dir = os.path.join(root_directory, 'test')
            image_dir = os.path.join(self.data_dir, 'test')
        else:
            raise Exception('No valid subset provided')

        # Create set to prevent redundant image_id's (string, partial file name)
        image_id_set = set()
        for root, dirs, filenames in os.walk(self.data_dir):
          for filename in filenames:
              image_id = getCoreImageFileName(filename)
              image_id_set.add(image_id)
        
        # Add unique image id's to dataset
        for image_id in image_id_set:
          city = image_id.split('_')[0] # First element in list should be city
          path = os.path.join(image_dir, city, image_id + '_leftImg8bit.png')
          self.add_image(source = "cityscape", 
          image_id=image_id,
          path=path)
            
        #print('---------------')
        #print(self._image_ids)
        #print('---------------')
        #print(len(self._image_ids))
        # return self._image_ids
        #image_id = random.choice(self._image_ids)
        #print("Sample image: %s" % image_id)

    def load_mask(self, image_id):
        '''
        Loads mask corresponding to an image id
        
        image_id: the unique id of the form city_sequenceNb_frame_Nb
        
        returns a bool array of masks and a list of class ids
        The polygons are extracted from the json files and constructed into a binary image
        using PIL. 
        '''
        
        # Retrieve available image metadata from dataset
        image_info = self.image_info[image_id]
        image_name = image_info['id']

        # Fetch and process the required metadata for the mask 
        city = image_name.split('_')[0] # First element in list should be city
        annotation_path = os.path.join(os.path.join(self.data_dir, city), image_name + '_gtFine_polygons.json')
        ann_dict = {}
        
        with open(annotation_path) as annotation:
            ann_dict = json.load(annotation)
        masks = []
        class_ids = []
        
        for obj in tqdm(ann_dict['objects']):
            # Must search list of dictionaries to find class_id (int) assosciated with class_name (string)
            class_name = obj['label']
            if class_name.endswith('group'):  # Some classes can be grouped if no clear boundary can be seen
              class_name = class_name[:-5]    # Remove group from the class name and continue as if one object
              #print('\nGroup removed from class %s\n' % class_name)
              
            #class_dict = next(item for item in self.class_info if item["name"] == class_name)
            class_dict = list(filter(lambda class_info_item: class_info_item['name'] == class_name, self.class_info))
            if (len(class_dict) == 0):
              print('Class %s not handled by current software\n' % class_name)
            else:
              class_ids.append(class_dict[0]['id'])

            # Generate bitmask skeleton for polygon drawing
            mask = Image.new(mode = '1', size = (ann_dict['imgWidth'], ann_dict['imgHeight']))
            draw = ImageDraw.Draw(mask)
            
            # Retrieve bitmask polygon info from JSON
            try:
                points = obj['polygon']
            except:
                print('no polygons for {}'.format(obj['label']))
            
            # PIL expects a tuple of tuples for points
            points = [tuple(coords) for coords in points]
            points = tuple(points)
            
            # Draw bitmask polygon from points
            draw.polygon((points), fill=1)
            masks.append(mask)

        if (class_ids):
          # Stack masks and class_ids
          masks = np.stack(masks, axis=2).astype(np.bool)
          class_ids = np.array(class_ids, dtype=np.int32)
          return masks, class_ids
        else:
          # Return empty mask
          return super(CityscapesSegmentationDataset, self).load_mask(image_id)