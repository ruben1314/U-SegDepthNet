from detect import inference_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def convert_channels_toRGB(image):
    image_RGB = np.zeros((192, 624, 3), dtype=np.int16)
    classes_rgb = [[210, 0, 200],  # Terrain
                   [90, 200, 255],  # Sky
                   [0, 199, 0],  # Tree
                   [90, 240, 0],  # Vegetation
                   [140, 140, 140],  # Building
                   [100, 60, 100],  # Road
                   [250, 100, 255],  # GuardRail
                   [255, 255, 0],  # TrafficSign
                   [200, 200, 0],  # TrafficLight
                   [255, 130, 0],  # Pole
                   [80, 80, 80],  # Misc
                   [160, 60, 60],  # Truck
                   [255, 127, 80],  # Car
                   [0, 139, 139],  # Van
                   [0, 0, 0]]  # Undefined
    classes = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Terrain
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sky
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Tree
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Vegetation
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Building
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Road
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # GuardRail
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # TrafficSign
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # TrafficLight
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Pole
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Misc
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Truck
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Car
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Van
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]  # Undefined
    for index_class, output_class in enumerate(classes, start=0):
        chequeo = np.where(np.all(image == output_class, axis=-1))
        if len(chequeo[0]) == 0:
            continue
        image_RGB[chequeo[0][:], chequeo[1][:]] = classes_rgb[index_class]
    return image_RGB


def extract_road(image_seg):
    image_mask = np.zeros((192, 624))
    chequeo = np.where(np.all(image_seg == [100, 60, 100], axis=-1))
    for i, j in zip(chequeo[0], chequeo[1]):
        image_mask[i, j] = 1
    kernel = np.ones((5,5),np.uint8)
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel)
    return image_mask


def extract_trayectory(road_mask, horizon=99):
    height = road_mask.shape[0]
    points = []
    for y_point in range(height-1, horizon, -1):
        print(y_point)
        road_points = np.where(road_mask[y_point] == 1)
        if len(road_points[0]) >= 1:
            avg = int(np.median(road_points[0]))
        points.append((y_point, avg))
    return points

def refine_points(trayectory_points):
    draw_points = []
    crop_lines = 20
    for point_index in range(0,len(trayectory_points), crop_lines):
        image_crop_points = trayectory_points[point_index:point_index+crop_lines]
        puntos = list(map(lambda x: x[1], image_crop_points))
        x_avg = int(np.average(puntos))
        draw_points.append((trayectory_points[point_index][0], x_avg))
    return draw_points

if __name__ == "__main__":
    with open("./datasets/test.txt", 'r') as f:
        lines = f.readlines()
        print("Loading dataset in: ", "./datasets/test.txt")
        horizon = 99
    image_counter = 0
    images_path = '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/'
    images_names = os.listdir(images_path)
    images_names = sorted(images_names)
    for depth_image_UNET, seg_image_UNET in inference_image('./models_ssh/mix_full/basicUNET_epoch52.torch', '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0'):
        image_seg_rgb = convert_channels_toRGB(seg_image_UNET)
        image_rgb = cv2.imread(images_path + images_names[image_counter])
        image_rgb = cv2.resize(image_rgb, (624, 192),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imshow("RGB", image_rgb)
        # cv2.waitKey(1)
        # plt.imshow(image_seg_rgb)
        # plt.waitforbuttonpress()
        road_mask = extract_road(image_seg_rgb)
        road_mask = np.resize(road_mask, (192, 624))
        # plt.imshow(road_mask)
        # plt.waitforbuttonpress()
        # road_mask = cv2.cvtColor(road_mask, cv2.COLOR_RGB2GRAY)
        trayectory_points = extract_trayectory(road_mask, horizon=horizon)
        trayectory_points = refine_points(trayectory_points)
        for point1, point2 in zip(trayectory_points[:], trayectory_points[1:]):
            # cv2.circle(image_rgb, (line_points[1], line_points[0]), radius=1,
            #            color=(255, 0, 0), thickness=5)
            cv2.line(image_rgb, (point1[1], point1[0]), (point2[1],point2[0]),
                     color=(255,0,0),thickness=3)
        cv2.imshow("trayectory", image_rgb)
        cv2.waitKey()
        # plt.waitforbuttonpress()
        image_counter += 1
