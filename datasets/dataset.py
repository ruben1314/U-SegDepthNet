import numpy as np
import os
import random
import cv2 as cv
import matplotlib.pyplot as plt
from time import time

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

    def get_out_channels(self):
        return self.output_classes
    
    def get_seg_channels(self):
        if self.output_classes == 16:
            return self.output_classes - 1
        return self.output_classes
    

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
        if os.path.isdir(base_dir):
            samples = os.listdir(base_dir)
            samples = sorted(samples)
        else:
            with open(base_dir , 'r') as f:
                samples = f.readlines()
        # print("samples", samples)
        # samples = sorted(samples)
        # if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        for sample in samples:
            # sample_dir = base_dir + os.path.sep + sample
            if os.path.isdir(base_dir):
                sample = base_dir + '/' + sample
            sample_dir = sample.replace('\n','')
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

        


class CityScapes():
    def __init__(self, data_dir="", batch_size=4, seg_image=False, depth_image=False, output_classes=19, save_npy=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seg_image = seg_image
        self.depth_image = False 
        self.output_classes = output_classes
        self.save_npy = save_npy
        self.name_classes = ["unlabelled",
                            "road",
                            "sidewalk",
                            "building",
                            "wall",
                            "fence",
                            "pole",
                            "traffic_light",
                            "traffic_sign",
                            "vegetation",
                            "terrain",
                            "sky",
                            "person",
                            "rider",
                            "car",
                            "truck",
                            "bus",
                            "train",
                            "motorcycle",
                            "bicycle",
                        ]
        self.colors = [  # [  0,   0,   0],
                    [128, 64, 128],
                    [244, 35, 232],
                    [70, 70, 70],
                    [102, 102, 156],
                    [190, 153, 153],
                    [153, 153, 153],
                    [250, 170, 30],
                    [220, 220, 0],
                    [107, 142, 35],
                    [152, 251, 152],
                    [0, 130, 180],
                    [220, 20, 60],
                    [255, 0, 0],
                    [0, 0, 142],
                    [0, 0, 70],
                    [0, 60, 100],
                    [0, 80, 100],
                    [0, 0, 230],
                    [119, 11, 32],
    ]

    def get_out_channels(self):
        return self.output_classes
    
    def get_seg_channels(self):
        return self.output_classes
    
    def load_train(self, train=True, shuffle=True, max_percent=1, print_images_load=100):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 192, 624),dtype=np.float16)
        targets = np.zeros((batch_size, self.output_classes, 192, 624), dtype=np.float32)
        if train:
            with open("./datasets/train_city.txt" , 'r') as f:
                lines = f.readlines()
                print("Loading dataset in: ", "./datasets/train_city.txt")
        else:
            with open("./datasets/test_city.txt" , 'r') as f:
                lines = f.readlines()
                print("Loading dataset in: ", "./datasets/test_city.txt")
        
        # samples = os.listdir(os.path.join(base_dir,"images"))
        
        # if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        class_by_image = dict()
        lines = lines[:int(len(lines)*max_percent)]
        lines_to_process = len(lines)
        for sample in lines:
            start_time = time()
            if self.seg_image:
                image_seg_path = sample
                image_seg_path = image_seg_path.replace("/leftImg8bit/", "/gtFine/")
                basename = os.path.basename(image_seg_path)
                image_seg_path = image_seg_path.replace(basename,"")
                file_name, ext = os.path.splitext(basename)
                file_name_splitted = file_name.split("_")
                # basename = "classgt_" + file_name_splitted[-1] + ".png"
                basename = file_name_splitted[0] + "_" + file_name_splitted[1] + "_" + file_name_splitted[2] + "_gtFine_color.npy"
                image_seg_path = os.path.join(image_seg_path,basename)
                # image_seg_path = image_seg_path.replace("_rgb/","_classSegmentation/")
                # print("Image seg path", image_seg_path)

            sample = sample.replace("\n","")
            time_imread = time()
            image = cv.imread(sample)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (624,192), interpolation=cv.INTER_NEAREST)
            # print("Tiempo imread rgb", time() - time_imread)
            if self.seg_image:
                # image_seg = cv.imread(image_seg_path)
                # image_seg = cv.cvtColor(image_seg, cv.COLOR_BGR2RGB)
                time_rgb_channels = time() 
                # image_seg_channels, class_by_image = self._get_image_seg_channels(image_seg, class_by_image)
                image_seg_channels = np.load(image_seg_path)
                image_seg_channels = np.moveaxis(image_seg_channels,1,3)
                image_seg_channels = image_seg_channels[0]
                # print("Time rgb channels", time() - time_rgb_channels)
                # image_seg_channels = cv.resize(image_seg_channels, (624,192), interpolation=cv.INTER_NEAREST)
            # print("Cargadas imagenes and converted", time() - start_time)

            # image_rgb = self.convert_channels_toRGB(image_seg_channels)
            # plt.figure(1)
            # plt.imshow(image)
            # plt.show()
            # plt.figure(2)
            # plt.imshow(image_seg_channels[:,:,12])
            # plt.figure(3)
            # plt.imshow(image_rgb)
            # plt.waitforbuttonpress()
            time_batches = time()
            batch[i%batch_size, 0] = self.norm(image[:,:,0])
            batch[i%batch_size, 1] = self.norm(image[:,:,1])
            batch[i%batch_size, 2] = self.norm(image[:,:,2])
            if self.seg_image:
                targets[i%batch_size, 0] = image_seg_channels[:,:,0]
                targets[i%batch_size, 1] = image_seg_channels[:,:,1]
                targets[i%batch_size, 2] = image_seg_channels[:,:,2]
                targets[i%batch_size, 3] = image_seg_channels[:,:,3]
                targets[i%batch_size, 4] = image_seg_channels[:,:,4]
                targets[i%batch_size, 5] = image_seg_channels[:,:,5]
                targets[i%batch_size, 6] = image_seg_channels[:,:,6]
                targets[i%batch_size, 7] = image_seg_channels[:,:,7]
                targets[i%batch_size, 8] = image_seg_channels[:,:,8]
                targets[i%batch_size, 9] = image_seg_channels[:,:,9]
                targets[i%batch_size, 10] = image_seg_channels[:,:,10]
                targets[i%batch_size, 11] = image_seg_channels[:,:,11]
                targets[i%batch_size, 12] = image_seg_channels[:,:,12]
                targets[i%batch_size, 13] = image_seg_channels[:,:,13]
                targets[i%batch_size, 14] = image_seg_channels[:,:,14]
            # print("Almacenado batch", time() - time_batches)

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
            if self.save_npy:
                image_seg_npy = image_seg_path.replace(".png",".npy")
                with open(image_seg_npy, 'wb') as f:
                    np.save(f, targets)
            # print("Tiempo de carga por imagen", time() - start_time)
            i += 1
            if ((i % print_images_load) == 0) or (i==1):
                print("Imagenes cargadas", i, "/", lines_to_process, "para Train =", train, "test =", not train)
            if i > 0 and (i%batch_size) == 0:
                yield batch, targets
        print("Clases por batches", class_by_image)


    def _get_image_seg_channels(self, image_ori, class_by_image):
        image_seg = np.zeros((image_ori.shape[0], image_ori.shape[1], len(self.colors)), dtype=np.int8)
        for index_class, class_seg in enumerate(self.colors,start=0):
            pixel_seg = np.zeros(len(self.colors))
            pixel_seg[index_class] = 1
            # chequeo = np.where(image_ori[:,:]==class_seg)
            time_where = time()
            chequeo = np.where(np.all(image_ori==class_seg, axis=-1))
            # print("Tiempo where", time() - time_where)
            image_seg[chequeo[0][:], chequeo[1][:]] = pixel_seg.astype(np.int8)
            if len(chequeo[0]) > 0:
                try:
                    class_by_image[self.name_classes[index_class]] += 1
                except:
                    class_by_image[self.name_classes[index_class]] = 1
        return image_seg, class_by_image

    def norm(self, image):
        ''' Normalize Image '''
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def load_test(self):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 192, 624))
        base_dir = self.data_dir
        print("Loading dataset in: ", self.data_dir)
        if os.path.isdir(base_dir):
            samples = os.listdir(base_dir)
            samples = sorted(samples)
        else:
            with open(base_dir , 'r') as f:
                samples = f.readlines()
        # print("samples", samples)
        # samples = sorted(samples)
        # if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        for sample in samples:
            # sample_dir = base_dir + os.path.sep + sample
            if os.path.isdir(base_dir):
                sample = base_dir + '/' + sample
            sample_dir = sample.replace('\n','')
            print("Imagen", sample_dir)
            image = cv.imread(sample_dir)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (624,192), interpolation=cv.INTER_NEAREST)
            # plt.imshow(image)
            # plt.show()
            batch[i%batch_size, 0] = self.norm(image[:,:,0])
            batch[i%batch_size, 1] = self.norm(image[:,:,1])
            batch[i%batch_size, 2] = self.norm(image[:,:,2])
            # Yield when batch is full
            i += 1
            if i > 0 and (i%batch_size) == 0:
                yield batch.copy()


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
        # for index_class, output_class in enumerate(classes, start=0):
        for index_class in range(0,19):
            output_class = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            output_class[index_class] = 1
            chequeo = np.where(np.all(image==output_class, axis=-1))
            if len(chequeo[0]) == 0:
                continue
            image_RGB[chequeo[0][:], chequeo[1][:]] = self.colors[index_class]
        return image_RGB