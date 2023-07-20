import sys
# sys.path.append('./monodepth2/')
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mtpi
import cv2
import numpy as np
from detect import inference_image
from torch.nn import MSELoss
import torch
from datasets.dataset import VirtualKitty, CityScapes
# os.chdir('./monodepth2/')
# from monodepth2.depth_prediction import predict_images
from depth_prediction import predict_images
os.chdir('./mmsegmentation/')

from mmseg.apis import inference_segmentor, init_segmentor
# import mmcv

config_file = 'configs/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes.py'
checkpoint_file = 'configs/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model_segmentator = init_segmentor(config_file, checkpoint_file, device='cuda:0') # 503 en vram
for name, parameter in model_segmentator.named_parameters():
    print("Parameter", name, parameter.numel())
pytorch_total_params = sum(p.numel() for p in model_segmentator.parameters() if p.requires_grad)
print("Numero de parametros totales", pytorch_total_params) # 194mb en vram
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
    counter = 0
    for frame_path in lines:    
        frame_path  = frame_path.replace('\n','')
        frame = cv2.imread(frame_path)
        result, final_seg_time = inference_segmentor(model_segmentator, frame)
        # image_name = os.path.basename(frame_path)
        os.makedirs('../compare_networks/mmsegmentation/', exist_ok=True)
        result_bgr = model_segmentator.show_result(frame, result, show=False, opacity=1)#, out_file='../compare_networks/mmsegmentation/'+str(counter)+'.jpg')
        # cv2.imshow("bgr", result_bgr)
        # cv2.waitKey()
        # print("Os actual dir", os.getcwd())
        cv2.imwrite('../compare_networks/mmsegmentation/' + str(counter)+'.jpg',result_bgr)
        counter += 1
        yield result_bgr, final_seg_time

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

def read_seg_gt_Vkitty(image_path):
    image_seg_path = image_path
    image_seg_path = image_seg_path.replace("/rgb/", "/classSegmentation/")
    basename = os.path.basename(image_seg_path)
    image_seg_path = image_seg_path.replace(basename,"")
    file_name, ext = os.path.splitext(basename)
    file_name_splitted = file_name.split("_")
    basename = "classgt_" + file_name_splitted[-1] + ".npy"
    image_seg_path = os.path.join(image_seg_path,basename)
    image_seg_path = image_seg_path.replace("_rgb/","_classSegmentation/")
    image_seg_channels = np.load(image_seg_path).astype(np.int8)
    image_seg_channels = np.moveaxis(image_seg_channels,1,3)
    image_seg_channels = image_seg_channels[0]
    return image_seg_channels[:,:,:15]

def read_seg_GT(image_path):
    # image_seg_path = image_path
    # image_seg_path = image_seg_path.replace("/rgb/", "/classSegmentation/")
    # basename = os.path.basename(image_seg_path)
    # image_seg_path = image_seg_path.replace(basename,"")
    # file_name, ext = os.path.splitext(basename)
    # file_name_splitted = file_name.split("_")
    # basename = "classgt_" + file_name_splitted[-1] + ".png"
    # image_seg_path = os.path.join(image_seg_path,basename)
    # image_seg_path = image_seg_path.replace("_rgb/","_classSegmentation/")
    # image_seg = cv2.imread(image_seg_path)
    # image_seg = cv2.cvtColor(image_seg, cv2.COLOR_BGR2RGB)
    # image_seg_channels = get_image_seg_channels(image_seg)
    # return cv2.resize(image_seg_channels, (624,192), interpolation=cv2.INTER_NEAREST)
    image_seg_path = image_path
    image_seg_path = image_seg_path.replace("/leftImg8bit/", "/gtFine/")
    basename = os.path.basename(image_seg_path)
    image_seg_path = image_seg_path.replace(basename,"")
    file_name, ext = os.path.splitext(basename)
    file_name_splitted = file_name.split("_")
    # basename = "classgt_" + file_name_splitted[-1] + ".png"
    basename = file_name_splitted[0] + "_" + file_name_splitted[1] + "_" + file_name_splitted[2] + "_gtFine_color.npy"
    image_seg_path = os.path.join(image_seg_path,basename)
    image_seg_channels = np.load(image_seg_path)
    image_seg_channels = np.moveaxis(image_seg_channels,1,3)
    return image_seg_channels[0].astype(np.int8)


def calculate_MSE_depth(imagen_monodepth, imagen_UNET, imagen_GT):
    mse_error = MSELoss()
    imagen_monodepth_tensor = torch.tensor(imagen_monodepth)
    imagen_GT_tensor = torch.tensor(imagen_GT)
    imagen_UNET_tensor = torch.tensor(imagen_UNET)
    error_monodepth = mse_error(imagen_monodepth_tensor, imagen_GT_tensor)
    error_UNET = mse_error(imagen_UNET_tensor, imagen_GT_tensor)
    print("min max monodepth", np.min(imagen_monodepth), np.max(imagen_monodepth), "min max unet", np.min(imagen_UNET), np.max(imagen_UNET))
    print("Error imagenes monodepth=", error_monodepth, "unet=", error_UNET)
    return error_monodepth, error_UNET

def calculate_IOU(imagen_seg_mmseg, seg_image_UNET, image_GT_seg):
    imagen_seg_mmseg_tensor = torch.tensor(imagen_seg_mmseg, dtype=torch.int8)
    seg_image_UNET_tensor = torch.tensor(seg_image_UNET)
    image_GT_seg_tensor = torch.tensor(image_GT_seg)
    # print("seg unet shape", seg_image_UNET_tensor.shape, seg_image_UNET_tensor.dtype, " image gt tensor shape", image_GT_seg_tensor.shape, "dtype", image_GT_seg_tensor.dtype, "mmseg", imagen_seg_mmseg_tensor.shape, imagen_seg_mmseg_tensor.dtype)
    iou_UNET = IOU(seg_image_UNET_tensor, image_GT_seg_tensor)
    iou_mmseg = IOU(imagen_seg_mmseg_tensor, image_GT_seg_tensor)
    print("Error UNET", iou_UNET, "mmseg", iou_mmseg)
    return iou_mmseg, iou_UNET

if __name__ == "__main__":
    with open("./datasets/test.txt" , 'r') as f:
        lines = f.readlines()
        print("Loading dataset in: ", "./datasets/test.txt")
    with open("./datasets/test_city.txt", "r") as f:
        lines_cityscapes = f.readlines()
    depth_monodepth_errors = []
    depth_UNET_errors = []
    depth_UNET_depth_errors =[]
    seg_mmseg_iou = []
    seg_UNET_iou = []
    seg_UNET_combined_iou = []
    seg_UNET_seg_iou =[]

    # os.chdir('./monodepth2/')
    # print(os.getcwd())
    # image = predict_images(lines)
    # os.chdir('./monodepth2/')
    gen_depth = predict_images(lines) # 14842236
    # os.chdir('../mmsegmentation/')
    gen_seg = get_imagen_segmentator(lines_cityscapes) # 48975494
    # os.chdir('..')
    gen_UNET = inference_image('/workspace/ruben/output_mix_full/basicUNET_epoch32.torch', './datasets/test.txt',VirtualKitty) 
    gen_UNET_segmentation = inference_image('/workspace/datastorage/model_seg_vkitty/basicUNET_epoch44.torch', './datasets/test.txt',VirtualKitty, 15)
    gen_UNET_depth = inference_image('/workspace/datastorage/model_depth_vkitty/basicUNET_epoch44.torch', './datasets/test.txt',VirtualKitty, 1)
    gen_UNET_city = inference_image('/workspace/datastorage/output_city_scapes/UNet.torch', './datasets/test_city.txt',CityScapes,19)
    cityScapes_loaded = CityScapes("",1,True, False,19, False)
    times_UNET = []
    times_unet_segmentation = []
    times_seg = []
    times_depth = []
    times_UNET_depth = []
    for image_number in range(len(lines)):
        image_GT_depth = read_depth_gt(lines[image_number])
        image_GT_seg = read_seg_gt_Vkitty(lines[image_number])
        os.chdir('./monodepth2/')
        # start_depth = time.time()
        imagen_depth,vmax, final_time_depth = next(gen_depth)
        # print("Tiempo depth", final_time_depth)
        # final_time_depth = time.time() - start_depth
        times_depth.append(final_time_depth)
        print("Media de tiempo de profundidad monodepth", np.average(times_depth))
        imagen_depth = cv2.resize(imagen_depth, (624,192), interpolation=cv2.INTER_NEAREST)
        # imagen_depth = imagen_depth * 65536
        # vmax = vmax * 65536
        # plt.imshow(imagen_depth, cmap='gray_r', vmax=vmax)
        # plt.waitforbuttonpress()
        os.chdir('..')
        print(os.getcwd())
        depth_image_UNET, seg_image_UNET, time_UNET_combined = next(gen_UNET) # 39394448 # 148mb en vram
        
        _, seg_image_UNET_segmentation, time_unet_segmentation = next(gen_UNET_segmentation) # 39394383 # 1078mb en vram
        # time.sleep(60)
        depth_image_UNET_depth, _, time_unet_depth = next(gen_UNET_depth) # 39393473 # 168mb en vram
        
        # print("Tiempo unet", time_UNET_combined)
        times_UNET.append(time_UNET_combined)
        times_unet_segmentation.append(time_unet_segmentation)
        times_UNET_depth.append(time_unet_depth)
        depth_image_UNET = cv2.resize(depth_image_UNET, (624,192), interpolation=cv2.INTER_NEAREST)
        depth_image_UNET_depth = cv2.resize(depth_image_UNET_depth, (624,192), interpolation=cv2.INTER_NEAREST)
        # seg_image_UNET = cv2.resize(seg_image_UNET, (624,192), interpolation=cv2.INTER_NEAREST)
        
        # Apply log transformation method
        # depth_image_UNET = depth_image_UNET * 65536
        # c = 65536 / np.log(1 + np.max(depth_image_UNET))
        # depth_image_UNET = c * (np.log(depth_image_UNET + 1))
        # plt.imshow(depth_image_UNET, cmap='gray')
        # plt.waitforbuttonpress()

        iou_UNET_combined, iou_UNET_segmentation = calculate_IOU(seg_image_UNET, seg_image_UNET_segmentation, image_GT_seg)
        seg_UNET_combined_iou.append(iou_UNET_combined)
        seg_UNET_seg_iou.append(iou_UNET_segmentation)
        error_monodepth, error_UNET = calculate_MSE_depth(imagen_depth, depth_image_UNET, image_GT_depth)
        error_UNET_depth, _ = calculate_MSE_depth(depth_image_UNET_depth, depth_image_UNET, image_GT_depth)
        depth_monodepth_errors.append(error_monodepth)
        depth_UNET_errors.append(error_UNET)
        depth_UNET_depth_errors.append(error_UNET_depth)
        print("Average times unet=", np.average(times_UNET), " unet_depth=", np.average(times_UNET_depth), " depth=", np.average(times_depth), "seg=", np.average(times_seg), "UNET_segmentation=", np.average(times_unet_segmentation))
        print("Imagen", image_number, "de", len(lines))
    
    os.makedirs('./runs/compareNetworks/', exist_ok=True)
    len_runs = len(os.listdir('./runs/compareNetworks/'))
    os.makedirs('./runs/compareNetworks/run'+str(len_runs+1), exist_ok=True)
    for image_number in range(len(lines_cityscapes)):
        image_GT_seg = read_seg_GT(lines_cityscapes[image_number])
        print("Path image", lines_cityscapes[image_number])
        os.chdir('./mmsegmentation')
        # start_seg_time = time.time()
        imagen_seg, final_seg_time = next(gen_seg)
        # print("Tiempo segmentatioon", final_seg_time)
        # final_seg_time = time.time() - start_seg_time
        times_seg.append(final_seg_time)
        # print("imagen seg shape", imagen_seg.shape)
        imagen_seg = cv2.resize(imagen_seg, (624,192), interpolation=cv2.INTER_NEAREST)
        imagen_seg = cv2.cvtColor(imagen_seg, cv2.COLOR_BGR2RGB)
        imagen_seg_channels,_ = cityScapes_loaded._get_image_seg_channels(imagen_seg, dict())
        # print("imagen", imagen_seg_channels)
        os.chdir('..')
        _, seg_image_UNET_city,_ = next(gen_UNET_city)
        seg_image_UNET_city = cv2.resize(seg_image_UNET_city, (624,192), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('./runs/compareNetworks/run'+str(len_runs+1)+'/'+str(image_number)+'.jpg', cityScapes_loaded.convert_channels_toRGB(seg_image_UNET_city))
        iou_mmseg, iou_UNET = calculate_IOU(imagen_seg_channels, seg_image_UNET_city, image_GT_seg)
        seg_UNET_iou.append(iou_UNET)
        seg_mmseg_iou.append(iou_mmseg)
        print("Imagen", image_number, "de", len(lines_cityscapes))
    print("Average depth error Monodepth2 = ", np.average(depth_monodepth_errors), "UNET=", np.average(depth_UNET_errors))
    print("Average depth error UNET_depth = ", np.average(depth_UNET_depth_errors), "UNET=", np.average(depth_UNET_errors))
    print("Average iou seg vs combined combined=", np.average(seg_UNET_combined_iou), "Segmentation=", np.average(seg_UNET_seg_iou))
    print("Average seg iou mmseg=", np.average(seg_mmseg_iou), "UNET=",np.average(seg_UNET_iou))
    print("Average times unet=", np.average(times_UNET), " depth=", np.average(times_depth), "seg=", np.average(times_seg), "UNET_segmentation=", np.average(times_unet_segmentation))
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



