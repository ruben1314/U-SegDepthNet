import numpy as np
import os
import random
import cv2 as cv

class VirtualKitty():
    
    def __init__(self, data_dir, batch_size=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.classes = [[210,0,200], #Terrain
                        [90,200,255], #Sky
                        [0,199,0], #Tree
                        [90,240,0], #Vegetation
                        [140,140,140], #Building
                        [100,60,100], #Road
                        [250,100,255], #GuardRail
                        [255,255,0], #TrafficSign
                        [200,200,0], #TrafficLight
                        [255,130,0], #Pole
                        [80,80,80], #Misc
                        [160,60,60], #Truck
                        [255,127,80], #Car
                        [0,139,139], #Van
                        [0,0,0]] # Undefined

    def norm(self, image):
        ''' Normalize Image '''
        return (image - np.min(image)) / (np.max(image) - np.min(image))
        
    def process(self, sample):
        ''' Resize sample to the given size '''
        # sample = np.transpose(sample, (2, 0, 1))
        return sample
        
    def load_train(self, train=True, shuffle=True):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 384, 1248))
        targets = np.zeros((batch_size, 15, 384, 1248))
        base_dir = self.data_dir
        if train:
            base_dir = self.data_dir + os.path.sep + "train/"
        else:
            base_dir = self.data_dir + os.path.sep + "val/"
        print("Loading dataset in: ", base_dir)
        samples = os.listdir(os.path.join(base_dir,"images"))
        if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        for sample in samples:
            sample_dir = os.path.join(base_dir,"images") + os.path.sep + sample
            sample_name, _ = os.path.splitext(sample)
            sample_name_split = sample_name.split("_")
            # print("Sample load train", sample_dir   )
            image = cv.imread(sample_dir)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = np.pad(image, ((0,9),(0,6),(0,0)), mode='constant', constant_values=0)
            image = np.resize(image, (image.shape[0], image.shape[1], image.shape[2]))
            image_seg = cv.imread(os.path.join(base_dir,"labels") + os.path.sep + "classgt_" + sample_name_split[1] + ".png")
            image_seg = cv.cvtColor(image_seg, cv.COLOR_BGR2RGB)
            image_seg = np.pad(image_seg, ((0,9),(0,6),(0,0)), mode='constant', constant_values=0)
            image_seg = np.resize(image_seg, (image.shape[0], image.shape[1], image.shape[2]))
            image_seg_channels = self._get_image_seg_channels(image_seg)
            batch[i%batch_size, 0] = self.norm(self.process(image[:,:,0]))
            batch[i%batch_size, 1] = self.norm(self.process(image[:,:,1]))
            batch[i%batch_size, 2] = self.norm(self.process(image[:,:,2]))

            targets[i%batch_size, 0] = self.process(image_seg_channels[:,:,0])
            targets[i%batch_size, 1] = self.process(image_seg_channels[:,:,1])
            targets[i%batch_size, 2] = self.process(image_seg_channels[:,:,2])
            targets[i%batch_size, 3] = self.process(image_seg_channels[:,:,3])
            targets[i%batch_size, 4] = self.process(image_seg_channels[:,:,4])
            targets[i%batch_size, 5] = self.process(image_seg_channels[:,:,5])
            targets[i%batch_size, 6] = self.process(image_seg_channels[:,:,6])
            targets[i%batch_size, 7] = self.process(image_seg_channels[:,:,7])
            targets[i%batch_size, 8] = self.process(image_seg_channels[:,:,8])
            targets[i%batch_size, 9] = self.process(image_seg_channels[:,:,9])
            targets[i%batch_size, 10] = self.process(image_seg_channels[:,:,10])
            targets[i%batch_size, 11] = self.process(image_seg_channels[:,:,11])
            targets[i%batch_size, 12] = self.process(image_seg_channels[:,:,12])
            targets[i%batch_size, 13] = self.process(image_seg_channels[:,:,13])
            targets[i%batch_size, 14] = self.process(image_seg_channels[:,:,14])
            # Yield when batch is full
            i += 1
            if i > 0 and (i%batch_size) == 0:
                yield batch, targets

    def load_test(self):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 384, 1248))
        base_dir = self.data_dir
        print("Loading dataset in: ", self.data_dir)
        samples = os.listdir(base_dir)
        # print("samples", samples)
        samples = sorted(samples)
        # if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        for sample in samples:
            sample_dir = base_dir + os.path.sep + sample
            print("Imagen", sample_dir)
            image = cv.imread(sample_dir)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = np.pad(image, ((0,9),(0,6),(0,0)), mode='constant', constant_values=0)
            image = np.resize(image, (image.shape[0], image.shape[1], image.shape[2]))
            batch[i%batch_size, 0] = self.norm(self.process(image[:,:,0]))
            batch[i%batch_size, 1] = self.norm(self.process(image[:,:,1]))
            batch[i%batch_size, 2] = self.norm(self.process(image[:,:,2]))
            # Yield when batch is full
            i += 1
            if i > 0 and (i%batch_size) == 0:
                yield batch.copy()

    def _get_image_seg_channels(self, image_ori):
        classes = [[210,0,200], #Terrain
                   [90,200,255], #Sky
                   [0,199,0], #Tree
                   [90,240,0], #Vegetation
                   [140,140,140], #Building
                   [100,60,100], #Road
                   [250,100,255], #GuardRail
                   [255,255,0], #TrafficSign
                   [200,200,0], #TrafficLight
                   [255,130,0], #Pole
                   [80,80,80], #Misc
                   [160,60,60], #Truck
                   [255,127,80], #Car
                   [0,139,139], #Van
                   [0,0,0]] # Undefined
        image_seg = np.zeros((image_ori.shape[0], image_ori.shape[1], len(classes)), dtype=np.int8)
        for index_class, class_seg in enumerate(classes,start=0):
            pixel_seg = np.zeros(len(classes))
            pixel_seg[index_class] = 1
            # chequeo = np.where(image_ori[:,:]==class_seg)
            chequeo = np.where(np.all(image_ori==class_seg, axis=-1))
            image_seg[chequeo[0][:], chequeo[1][:]] = pixel_seg.astype(np.int8)
        return image_seg
    
    def convert_channels_toRGB(self, image):
        image_RGB = np.zeros((384, 1248,3))
        classes = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #Terrain
                   [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0], #Sky
                   [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0], #Tree
                   [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0], #Vegetation
                   [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], #Building
                   [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0], #Road
                   [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], #GuardRail
                   [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], #TrafficSign
                   [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0], #TrafficLight
                   [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0], #Pole
                   [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0], #Misc
                   [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], #Truck
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0], #Car
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0], #Van
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]] # Undefined
        for index_class, output_class in enumerate(classes, start=0):
            # print("output clases", len(output_class), output_class)
            # print("Pixel image", image[250,200], image[:,:].shape)
            chequeo = np.where(np.all(image==output_class, axis=-1))
            # print("chequeo", len(chequeo))
            if len(chequeo[0]) == 0:
                continue
            # if len(chequeo[0]) != 0 and index_class != 14:
                # print("clase distinta undefined", output_class, chequeo)
            image_RGB[chequeo[0][:], chequeo[1][:]] = self.classes[index_class]
            # print("Max image ", image_RGB.max())
        return image_RGB

        
