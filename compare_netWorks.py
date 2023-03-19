import sys
# sys.path.append('./monodepth2/')
import os
import matplotlib.pyplot as plt
import matplotlib.image as mtpi
import cv2
import numpy as np
from detect import inference_image
from torch.nn import MSELoss
import torch
os.chdir('./monodepth2/')
from monodepth2.depth_prediction import predict_images
os.chdir('../mmsegmentation/')

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model_segmentator = init_segmentor(config_file, checkpoint_file, device='cuda:0')
os.chdir('..')

def norm(image):
    ''' Normalize Image '''
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def IOU(preds, targets, smooth=0.001):
    # preds = preds.view(-1)
    # targets = targets.view(-1)
    # Intersection is equivalent to True Positive count
    # Union is the mutually inclusive area of all labels & predictions 
    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()
    # Compute Score
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

def get_image_seg_channels(image_ori):
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

def get_imagen_segmentator(lines):
    for frame_path in lines:    
        frame_path  = frame_path.replace('\n','')
        frame = cv2.imread(frame_path)
        result = inference_segmentor(model_segmentator, frame)
        image_name = os.path.basename(frame_path)
        os.makedirs('../compare_networks/mmsegmentation/', exist_ok=True)
        result_bgr = model_segmentator.show_result(frame, result, show=False, opacity=1)
        # cv2.imshow("bgr", result_bgr)
        # cv2.waitKey()
        yield result_bgr

def read_depth_gt(image_path):
    image_depth_path = image_path
    image_depth_path = image_depth_path.replace("/rgb/", "/depth/")
    basename = os.path.basename(image_depth_path)
    image_depth_path = image_depth_path.replace(basename,"")
    file_name, ext = os.path.splitext(basename)
    file_name_splitted = file_name.split("_")
    basename = "depth_" + file_name_splitted[-1] + ".png"
    image_depth_path = os.path.join(image_depth_path,basename)
    image_depth_path = image_depth_path.replace("_rgb/","_depth/")
    image_depth= cv2.imread(image_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH )
    image_depth = np.float32(image_depth)
    image_depth = cv2.resize(image_depth, (624,192), interpolation=cv2.INTER_NEAREST)
    return norm(image_depth)#| cv2.CV_32F)

def read_seg_GT(image_path):
    image_seg_path = image_path
    image_seg_path = image_seg_path.replace("/rgb/", "/classSegmentation/")
    basename = os.path.basename(image_seg_path)
    image_seg_path = image_seg_path.replace(basename,"")
    file_name, ext = os.path.splitext(basename)
    file_name_splitted = file_name.split("_")
    basename = "classgt_" + file_name_splitted[-1] + ".png"
    image_seg_path = os.path.join(image_seg_path,basename)
    image_seg_path = image_seg_path.replace("_rgb/","_classSegmentation/")
    image_seg = cv2.imread(image_seg_path)
    image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGR2RGB)
    image_seg_channels = get_image_seg_channels(image_seg)
    return cv2.resize(image_seg_channels, (624,192), interpolation=cv2.INTER_NEAREST)

def calculate_MSE_depth(imagen_monodepth, imagen_UNET, imagen_GT):
    mse_error = MSELoss()
    imagen_monodepth_tensor = torch.tensor(imagen_monodepth)
    imagen_GT_tensor = torch.tensor(imagen_GT)
    imagen_UNET_tensor = torch.tensor(imagen_UNET)
    error_monodepth = mse_error(imagen_monodepth_tensor, imagen_GT_tensor)
    error_UNET = mse_error(imagen_UNET_tensor, imagen_GT_tensor)
    print("Error imagenes monodepth=", error_monodepth, "unet=", error_UNET)
    return error_monodepth, error_UNET

def calculate_IOU(imagen_seg_mmseg, seg_image_UNET, image_GT_seg):
    imagen_seg_mmseg_tensor = torch.tensor(imagen_seg_mmseg)
    seg_image_UNET_tensor = torch.tensor(seg_image_UNET)
    image_GT_seg_tensor = torch.tensor(image_GT_seg)
    iou_UNET = IOU(seg_image_UNET_tensor, image_GT_seg_tensor)
    
    return 0, iou_UNET

if __name__ == "__main__":
    with open("./datasets/test.txt" , 'r') as f:
        lines = f.readlines()
        print("Loading dataset in: ", "./datasets/test.txt")
    depth_monodepth_errors = []
    depth_UNET_errors = []
    seg_mmseg_iou = []
    seg_UNET_iou = []
    # os.chdir('./monodepth2/')
    # print(os.getcwd())
    # image = predict_images(lines)
    # os.chdir('./monodepth2/')
    gen_depth = predict_images(lines)
    # os.chdir('../mmsegmentation/')
    gen_seg = get_imagen_segmentator(lines)
    # os.chdir('..')
    gen_UNET = inference_image('./models_ssh/mix_full/basicUNET_epoch10.torch')
    for image_number in range(len(lines)):
        image_GT_depth = read_depth_gt(lines[image_number])
        image_GT_seg = read_seg_GT(lines[image_number])
        os.chdir('./monodepth2/')
        imagen_depth,vmax = next(gen_depth)
        imagen_depth = cv2.resize(imagen_depth, (624,192), interpolation=cv2.INTER_NEAREST)
        # imagen_depth = imagen_depth * 65536
        # vmax = vmax * 65536
        os.chdir('../mmsegmentation')
        imagen_seg = next(gen_seg)
        imagen_seg = cv2.resize(imagen_seg, (624,192), interpolation=cv2.INTER_NEAREST)
        print("imagen")
        # plt.imshow(imagen_depth, cmap='gray_r', vmax=vmax)
        # plt.waitforbuttonpress()
        os.chdir('..')
        print(os.getcwd())
        depth_image_UNET, seg_image_UNET = next(gen_UNET)
        depth_image_UNET = cv2.resize(depth_image_UNET, (624,192), interpolation=cv2.INTER_NEAREST)
        seg_image_UNET = cv2.resize(seg_image_UNET, (624,192), interpolation=cv2.INTER_NEAREST)
        # Apply log transformation method
        # depth_image_UNET = depth_image_UNET * 65536
        # c = 65536 / np.log(1 + np.max(depth_image_UNET))
        # depth_image_UNET = c * (np.log(depth_image_UNET + 1))
        # plt.imshow(depth_image_UNET, cmap='gray')
        # plt.waitforbuttonpress()
        error_monodepth, error_UNET = calculate_MSE_depth(imagen_depth, depth_image_UNET, image_GT_depth)
        depth_monodepth_errors.append(error_monodepth)
        depth_UNET_errors.append(error_UNET)
        iou_mmseg, iou_UNET = calculate_IOU(imagen_seg, seg_image_UNET, image_GT_seg)
        seg_UNET_iou.append(iou_UNET)
        seg_mmseg_iou.append(iou_mmseg)
    print("Average depth error Monodepth2 = ", np.average(depth_monodepth_errors), "UNET=", np.average(depth_UNET_errors))
    print("Average seg iou mmseg=", np.average(seg_mmseg_iou), "UNET=",np.average(seg_UNET_iou))
    # imagen_counter = 0
    # for image, vmax in predict_images(lines):
    #     # image = np.subtract(1, image)
    #     # plt.imshow(image, cmap='gray_r', vmax=vmax)
    #     os.makedirs('../compare_networks/monodepth2/', exist_ok=True)
    #     image_name = os.path.basename(lines[imagen_counter])
    #     image_name = image_name.replace('\n','')
    #     mtpi.imsave('../compare_networks/monodepth2/'+image_name , image, vmax=vmax, cmap='gray_r')
    #     # plt.waitforbuttonpress()
    #     imagen_counter += 1
    
    # os.chdir('../mmsegmentation')

    # from mmseg.apis import inference_segmentor, init_segmentor
    # import mmcv

    # config_file = 'pspnet_r50-d8_512x1024_40k_cityscapes.py'
    # checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

    # # build the model from a config file and a checkpoint file
    # model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    # for frame_path in lines:
    #     frame_path  = frame_path.replace('\n','')
    #     frame = cv2.imread(frame_path)
    #     result = inference_segmentor(model, frame)
    #     image_name = os.path.basename(frame_path)
    #     os.makedirs('../compare_networks/mmsegmentation/', exist_ok=True)
    #     model.show_result(frame, result, show=False, opacity=1, out_file='../compare_networks/mmsegmentation/'+image_name)



