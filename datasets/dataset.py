import numpy as np
import os
import random
import cv2 as cv
import matplotlib.pyplot as plt

class VirtualKitty():
    
    def __init__(self, data_dir="", batch_size=4, seg_image=False, depth_image=False, output_classes=16):
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
        self.name_classes = ["Terrain",
                             "Sky",
                             "Tree",
                             "Vegetation",
                            "Building",
                            "Road",
                            "GuardRail",
                            "TrafficSign",
                            "TrafficLight",
                            "Pole",
                            "Misc",
                            "Truck",
                            "Car",
                            "Van",
                            "Undefined"]
        self.seg_image = seg_image
        self.depth_image = depth_image
        self.output_classes = output_classes

    def norm(self, image):
        ''' Normalize Image '''
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def convert_range_image(self, image):
        # return image * (pow(2,16) -1)
        return image * 255
        
    def process(self, sample):
        ''' Resize sample to the given size '''
        # sample = np.transpose(sample, (2, 0, 1))
        return sample
        
    def load_train(self, train=True, shuffle=True, max_percent=1, print_images_load=100):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 192, 624),dtype=np.float16)
        targets = np.zeros((batch_size, self.output_classes, 192, 624), dtype=np.float32)
        if train:
            with open("./datasets/train.txt" , 'r') as f:
                lines = f.readlines()
                print("Loading dataset in: ", "./datasets/train.txt")
        else:
            with open("./datasets/test.txt" , 'r') as f:
                lines = f.readlines()
                print("Loading dataset in: ", "./datasets/test.txt")
        
        # samples = os.listdir(os.path.join(base_dir,"images"))
        
        # if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        class_by_image = dict()
        lines = lines[:int(len(lines)*max_percent)]
        lines_to_process = len(lines)
        for sample in lines:
            if self.seg_image:
                image_seg_path = sample
                image_seg_path = image_seg_path.replace("/rgb/", "/classSegmentation/")
                basename = os.path.basename(image_seg_path)
                image_seg_path = image_seg_path.replace(basename,"")
                file_name, ext = os.path.splitext(basename)
                file_name_splitted = file_name.split("_")
                basename = "classgt_" + file_name_splitted[-1] + ".png"
                image_seg_path = os.path.join(image_seg_path,basename)
                image_seg_path = image_seg_path.replace("_rgb/","_classSegmentation/")
            sample = sample.replace("\n","")
            if self.depth_image:
                image_depth_path = sample
                image_depth_path = image_depth_path.replace("/rgb/", "/depth/")
                basename = os.path.basename(image_depth_path)
                image_depth_path = image_depth_path.replace(basename,"")
                file_name, ext = os.path.splitext(basename)
                file_name_splitted = file_name.split("_")
                basename = "depth_" + file_name_splitted[-1] + ".png"
                image_depth_path = os.path.join(image_depth_path,basename)
                image_depth_path = image_depth_path.replace("_rgb/","_depth/")

            image = cv.imread(sample)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (624,192), interpolation=cv.INTER_NEAREST)
            if self.seg_image:
                image_seg = cv.imread(image_seg_path)
                image_seg = cv.cvtColor(image_seg, cv.COLOR_BGR2RGB)
                image_seg_channels, class_by_image = self._get_image_seg_channels(image_seg, class_by_image)
                image_seg_channels = cv.resize(image_seg_channels, (624,192), interpolation=cv.INTER_NEAREST)
            if self.depth_image:
                image_depth = cv.imread(image_depth_path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
                image_depth = cv.resize(image_depth, (624,192), interpolation=cv.INTER_NEAREST)
            # print("image_depth min", np.min(image_depth), "max", np.max(image_depth))
            # print("image depth", image_depth.dtype)
            # plt.imshow(image_depth, cmap='gray')
            # plt.waitforbuttonpress()

            # image_rgb = self.convert_channels_toRGB(image_seg_channels)
            # plt.figure(1)
            # plt.imshow(image)
            # plt.show()
            # plt.figure(2)
            # plt.imshow(image_seg_channels[:,:,12])
            # plt.figure(3)
            # plt.imshow(image_rgb)
            # plt.waitforbuttonpress()
            batch[i%batch_size, 0] = self.norm(self.process(image[:,:,0]))
            batch[i%batch_size, 1] = self.norm(self.process(image[:,:,1]))
            batch[i%batch_size, 2] = self.norm(self.process(image[:,:,2]))
            if self.seg_image:
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
            if self.depth_image:
                targets[i%batch_size, -1] = self.norm(image_depth[:,:])

            # print("Targets", np.min(targets[i%batch_size,15]), "max", np.max(targets[i%batch_size,15]), np.average(targets[i%batch_size, 15]))
            ####### Plot images input to debug errors
            # plt.subplot(6,3,1)
            # # print("image", batch[image_batch])
            # plt.imshow(image)
            # for channel_index in range(15):
            #     # print("channel", channel.shape)
            #     plt.subplot(6,3,channel_index+2)
            #     plt.imshow(image_seg_channels[:,:,channel_index])
            # plt.subplot(6,3,17)
            # plt.imshow(image_rgb)
            # plt.show()
            
            # plt.waitforbuttonpress()
            # Yield when batch is full
            i += 1
            if ((i % print_images_load) == 0) or (i==1):
                print("Imagenes cargadas", i, "/", lines_to_process, "para Train =", train, "test =", not train)
            if i > 0 and (i%batch_size) == 0:
                yield batch, targets
        print("Clases por batches", class_by_image)


    def load_test(self):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 192, 624))
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
            image = cv.resize(image, (624,192), interpolation=cv.INTER_NEAREST)
            # plt.imshow(image)
            # plt.show()
            batch[i%batch_size, 0] = self.norm(self.process(image[:,:,0]))
            batch[i%batch_size, 1] = self.norm(self.process(image[:,:,1]))
            batch[i%batch_size, 2] = self.norm(self.process(image[:,:,2]))
            # Yield when batch is full
            i += 1
            if i > 0 and (i%batch_size) == 0:
                yield batch.copy()

    def _get_image_seg_channels(self, image_ori, class_by_image):
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
            if len(chequeo[0]) > 0:
                try:
                    class_by_image[self.name_classes[index_class]] += 1
                except:
                    class_by_image[self.name_classes[index_class]] = 1
        return image_seg, class_by_image
    
    def convert_channels_toRGB(self, image):
        image_RGB = np.zeros((192, 624, 3), dtype=np.int16)
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
            chequeo = np.where(np.all(image==output_class, axis=-1))
            if len(chequeo[0]) == 0:
                continue
            image_RGB[chequeo[0][:], chequeo[1][:]] = self.classes[index_class]
        return image_RGB

        
