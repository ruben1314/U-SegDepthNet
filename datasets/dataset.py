import numpy as np
import os
import random
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib

class VirtualKitty():
    
    def __init__(self, data_dir, batch_size=4):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def norm(self, image):
        ''' Normalize Image '''
        return (image - np.min(image)) / (np.max(image) - np.min(image))
        
    def process(self, sample):
        ''' Resize sample to the given size '''
        # sample = np.transpose(sample, (2, 0, 1))
        return sample
        
    def load(self, shuffle=True):
        batch_size = self.batch_size
        # Init
        batch = np.zeros((batch_size, 3, 384, 1248))
        targets = np.zeros((batch_size, 15, 384, 1248))
        base_dir = self.data_dir
        print("Loading dataset in: ", self.data_dir)
        samples = os.listdir(os.path.join(base_dir,"images"))
        if shuffle: random.shuffle(samples)
        # Yield samples when batch is full
        i = 0
        for sample in samples:
            sample_dir = os.path.join(base_dir,"images") + os.path.sep + sample
            sample_name, _ = os.path.splitext(sample)
            # print("sample name", sample_name)
            sample_name_split = sample_name.split("_")
            # si = slice_idx
            # if slice_idx == 'random': 
            #     si = random.randint(90, 110)
            # Read samples
            # t1w = nib.load(sample_dir + os.path.sep + f"{sample}_t1.nii").get_fdata().astype('float32')
            # t1wpc = nib.load(sample_dir + os.path.sep + f"{sample}_t1ce.nii").get_fdata().astype('float32')
            # t2w = nib.load(sample_dir + os.path.sep + f"{sample}_t2.nii").get_fdata().astype('float32')
            # flair = nib.load(sample_dir + os.path.sep + f"{sample}_flair.nii").get_fdata().astype('float32')
            # seg = nib.load(sample_dir + os.path.sep + f"{sample}_seg.nii").get_fdata().astype('float32')
            image = cv.imread(sample_dir)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # image = np.zeros((image_read.shape[0]+1, image_read.shape[1], image_read.shape[2]))
            # image = image_read[:,:,:]
            # print("image_read shape antes padding", image.shape)
            # print("image shape antes padding", image.shape)
            image = np.pad(image, ((0,9),(0,6),(0,0)), mode='constant', constant_values=0)
            image = np.resize(image, (image.shape[0], image.shape[1], image.shape[2]))
            # image = image[:,:-2]
            # print("Image shape después del padding", image.shape)
            # image =
            image_seg = cv.imread(os.path.join(base_dir,"labels") + os.path.sep + "classgt_" + sample_name_split[1] + ".png")
            image_seg = cv.cvtColor(image_seg, cv.COLOR_BGR2RGB)
            image_seg = np.pad(image_seg, ((0,9),(0,6),(0,0)), mode='constant', constant_values=0)
            image_seg = np.resize(image_seg, (image.shape[0], image.shape[1], image.shape[2]))
            # image_seg = image_seg[:,:-2]
            # print("Shape image_seg", image_seg.shape)
            image_seg_channels = self._get_image_seg_channels(image_seg)
            # print("image_seg shape", image_seg.shape)
            # cv.imshow("image", image)
            # cv.waitKey(0)
            # Add to batch
            # Add to batch
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
            # batch[i%batch_size, 6] = self.norm(self.process(flair, si))
            # targets[i%batch_size, 0] = self.process(seg, si) 
            # Yield when batch is full
            i += 1
            if i > 0 and (i%batch_size) == 0:
                yield batch, targets


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
        # print("Image seg shape", image_seg.shape)
        for index_class, class_seg in enumerate(classes,start=0):
            pixel_seg = np.zeros(len(classes))
            pixel_seg[index_class] = 1
            # print("Image original", image_ori.shape)
            # print("Image seg", image_seg.shape)
            # print("Class seg", class_seg)
            # print("Pixel seg", pixel_seg.shape)
            # print("Pixel image", image_ori[175, 1116])
            # print("Check shape", (image_ori[:,:,:]==class_seg).shape)
            # print(image[:,:] == class_seg)
            chequeo = np.where(image_ori[:,:]==class_seg)
            # print("Pixel original", image_ori[chequeo[0][0], chequeo[1][0]])
            # print("Pixel copiar", chequeo[0][0], chequeo[1][0])
            # print(np.where(image_ori[:,:]==class_seg))
            image_seg[chequeo[0][:], chequeo[1][:]] = pixel_seg.astype(np.int8)
        # print("Image seg modificada", image_seg)
        # matplotlib.use('TKAgg', force=True)
        # for index in range(len(classes)):
        #     print("Clase mostrando", index)
        #     print("Shape image", image_seg[:,:,index].max())
        #     plt.imshow(image_seg[:,:,index], cmap='gray')
        #     plt.waitforbuttonpress()
        #     # cv.imshow("mask", image_seg[:,:,index])
        #     # cv.waitKey(0)
        return image_seg
