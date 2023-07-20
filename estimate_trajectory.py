from detect import inference_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from datasets.dataset import VirtualKitty

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
    image_mask = np.zeros((192, 624), np.uint8)
    chequeo = np.where(np.all(image_seg == [100, 60, 100], axis=-1))
    for i, j in zip(chequeo[0], chequeo[1]):
        image_mask[i, j] = 1
    cv2.imshow("Mask sin erosionar", image_mask*255)
    kernel_dilate = np.ones((5,5))
    image_mask = cv2.dilate(image_mask, kernel_dilate)
    kernel = np.ones((11,11),np.uint8)
    # image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, kernel)
    image_mask = cv2.erode(image_mask, kernel)
    kernel = np.ones((5,5),np.uint8)
    image_mask = cv2.erode(image_mask, kernel)
    image_mask = cv2.erode(image_mask, kernel)
    return image_mask


def extract_trajectory(road_mask, horizon=99):
    height = road_mask.shape[0]
    points = []
    for y_point in range(height-1, horizon, -1):
        # print(y_point)
        road_points = np.where(road_mask[y_point] == 1)
        if len(road_points[0]) >= 1:
            avg = int(np.median(road_points[0]))
            # avg = int(np.mean([min(road_points[0]), max(road_points[0])]))
            points.append((y_point, avg))
    return points

def refine_points(trajectory_points, crop_lines=20):
    draw_points = []
    # crop_lines = 20
    for point_index in range(0,len(trajectory_points), crop_lines):
        image_crop_points = trajectory_points[point_index:point_index+crop_lines]
        puntos = list(map(lambda x: x[1], image_crop_points))
        x_avg = int(np.average(puntos))
        draw_points.append((trajectory_points[point_index][0], x_avg))
    return draw_points

# def get_birdeye_image(rgb_image, seg_image, depth_image):
#     road_mask = extract_road(seg_image)
#     road_mask = np.resize(road_mask, (192, 624))
#     contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     contour_index = -1 
#     contour_points = 0
#     for index, contour in enumerate(contours,start=0):
#         if contour_points < contour.shape[0]:
#             contour_index = index
#             contour_points = contour.shape[0]
#     for contour in contours[contour_index]:
#         print("contour 0 ", contour[0][0])
#     image_rgb_paint = rgb_image.copy()
#     cv2.drawContours(image_rgb_paint, contours, -1, (0, 255, 0), 3)
#     cv2.imshow("Image rgb", image_rgb_paint)
#     cv2.waitKey()
#     image = np.zeros ((192,624,3), dtype=np.int16)
#     np_where = np.where(depth_image < (5000 / 65536))
#     image[np_where[0], np_where[1]] = seg_image[np_where[0], np_where[1]]
#     # plt.imshow(image)
#     # plt.waitforbuttonpress()
#     # cv2.imshow("where", image)
#     # cv2.waitKey()
#     max_depth = 5000/65536
#     birds_eye_image = np.zeros((192,624,3), dtype=np.int16)
#     for x,y in zip(np_where[0],np_where[1]):
#         depth_value = np.mean(depth_image[max(x-2,0):x+2,max(y-2,0):y+2])
#         if depth_value < max_depth:
#             x_bird = int((depth_value * 192) / max_depth)
#             birds_eye_image[x_bird, y] = seg_image[x,y]
            
#     # plt.imshow(birds_eye_image)
#     # plt.waitforbuttonpress()
#     pass


def get_transform_matrix(seg_image):
    IMAGE_H = 192
    IMAGE_W = 624
    seg_image_road = extract_road(seg_image)
    plt.imshow(seg_image_road)
    line = np.where(seg_image_road[191,:] == 1)
    top_road = np.where(seg_image_road[:,:] == 1)
    min_top = min(top_road[0])
    section_road = np.where(seg_image_road[min_top:min_top+15,:])
    
    src = np.float32([[min(line[0]), IMAGE_H], [max(line[0]), IMAGE_H], [ min(section_road[1]), min_top], [max(section_road[1]), min_top]])
    dst = np.float32([[250, IMAGE_H], [350, IMAGE_H], [250, 0], [350, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    return M, src

def get_birdeye_image(rgb_image, seg_image, depth_image):
    IMAGE_H = 192
    IMAGE_W = 624
    seg_image_road = extract_road(seg_image)
    # img = cv2.imread('./test_img.jpg') # Read the test img
    # img = seg_image_road[200:(200+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(seg_image_road, M, (IMAGE_W, IMAGE_H)) # Image warping
    warped_img_rgb = cv2.warpPerspective(rgb_image, M, (IMAGE_W, IMAGE_H)) # Image warping
    warped_img_seg_rgb = cv2.warpPerspective(seg_image, M, (IMAGE_W, IMAGE_H)) # Image warping
    # plt.imshow(seg_image_road)
    # plt.show()
    # plt.imshow(warped_img) # Show results
    # plt.show()
    return warped_img, warped_img_rgb, warped_img_seg_rgb
    
    
def get_cross(birdeye_image):
    pixels_road = np.where(birdeye_image[:,:] == 1)
    min_x, max_x = min(pixels_road[1]), max(pixels_road[1])
    if abs(min_x-250) > 50 and abs(max_x - 350) > 50:
        return True
    return False
    # corner_left = np.where(birdeye_image[0:100,0:50] == 1)
    # corner_right = np.where(birdeye_image[0:50, 524:624] == 1)
    # if len(corner_left[0]) > (0.25 * (100*50)) and len(corner_right[0]) > (0.25 * (100*50)):
    #     return True
    # return False

if __name__ == "__main__":
    with open("./datasets/test.txt", 'r') as f:
        lines = f.readlines()
        print("Loading dataset in: ", "./datasets/test.txt")
        
    image_counter = 0
    images_path = '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/'
    images_names = os.listdir(images_path)
    images_names = sorted(images_names)
    cross_frames = 0
    get_matrix_bool = True
    virtual_kitty = VirtualKitty("", 1)
    # for depth_image_UNET, seg_image_UNET, time_exec in inference_image('./models_ssh/mix_full/basicUNET_epoch32_combine.torch', '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0',VirtualKitty):
    for depth_image_UNET, seg_image_UNET, time_exec in inference_image('./models_ssh/mix_full/basicUNET_epoch100_combinetest.torch', '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0',VirtualKitty):
    # for depth_image_UNET, seg_image_UNET, time_exec in inference_image('./models_ssh/segbasicUNET_epoch44.torch', '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0',VirtualKitty):
    # for depth_image_UNET, seg_image_UNET, time_exec in inference_image('./models_ssh/depth_basicUNET_epoch44.torch', '../Dataset/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0',VirtualKitty):
        depth_rep = depth_image_UNET*255
        # depth_rep = (depth_image_UNET - np.min(depth_image_UNET)) / (np.max(depth_image_UNET) - np.min(depth_image_UNET)) * 255
        c = 255 / np.log(1 + np.max(depth_rep))
        depth_rep = c * (np.log(depth_rep + 1))
        # depth_rep = np.array(depth_rep, dtype=np.uint8)
        cv2.imshow("Depth", depth_rep)
        cv2.waitKey()
        continue
        image_seg_rgb = convert_channels_toRGB(seg_image_UNET)
        image_rgb = cv2.imread(images_path + images_names[image_counter])
        image_rgb = cv2.resize(image_rgb, (624, 192),
                               interpolation=cv2.INTER_NEAREST)
        if get_matrix_bool:
            get_matrix_bool = False
            M, points = get_transform_matrix(image_seg_rgb)
        image_rgb_points = image_rgb.copy()
        
        for point in points:
            # print("Point", point)
            cv2.circle(image_rgb_points, (int(point[0]),int(point[1])),2,(0,0,255), 5)
        
        cv2.imshow("RGB", image_rgb)
        cv2.imshow("Depth", depth_image_UNET*255)
        road_mask = extract_road(image_seg_rgb)
        road_mask = np.resize(road_mask, (192, 624))
        road_mask_points = road_mask.copy()
        for point in points:
            # print("Point", point)
            cv2.circle(road_mask_points, (int(point[0]),int(point[1])),2,(0,0,255), 5)
        birdeye_image, birdeye_rgb, warped_img_seg_rgb = get_birdeye_image(image_rgb, image_seg_rgb, depth_image_UNET)
        cross_bool = get_cross(birdeye_image)
        print("Cruce ", cross_bool)
        if cross_bool:
            cross_frames += 1
        else:
            cross_frames -= 1
            cross_frames = max(0, cross_frames)
        if cross_bool and (cross_frames > 4):
            cv2.putText(image_rgb, "Cruce", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            # plt.imshow(birdeye_image)
            # plt.waitforbuttonpress()
            # plt.imshow(road_mask)
            # plt.waitforbuttonpress()
        cv2.imshow("RGB", image_rgb)
        cv2.imshow("mask road", road_mask*255)
        cv2.imshow("birdeye", birdeye_image*255)
        cv2.imshow("Birdeye rgb", birdeye_rgb*255)
        cv2.imshow("RGB points", image_rgb_points)
        cv2.imshow("RGB SEG", warped_img_seg_rgb*255)
        cv2.imshow("Mask points", road_mask_points*255)
        # image_BGR = cv2.cvtColor(np.float32(image_seg_rgb), cv2.COLOR_RGB2BGR)
        image_seg_rgb = image_seg_rgb.astype(np.uint8 )
        cv2.imshow("Segmentation", image_seg_rgb)
        # cv2.waitKey()
        # plt.imshow(image_seg_rgb)
        # plt.waitforbuttonpress()
        
        # plt.imshow(road_mask)
        # plt.waitforbuttonpress()
        # road_mask = cv2.cvtColor(road_mask, cv2.COLOR_RGB2GRAY)
        horizon = 110
        crop_lines = 20
        if cross_bool:
            horizon = 140
            crop_lines = 5
        
        trajectory_points = extract_trajectory(road_mask, horizon=horizon)
        trajectory_points = refine_points(trajectory_points,crop_lines)
        for point1, point2 in zip(trajectory_points[:], trajectory_points[1:]):
            # cv2.circle(image_rgb, (line_points[1], line_points[0]), radius=1,
            #            color=(255, 0, 0), thickness=5)
            cv2.line(image_rgb, (point1[1], point1[0]), (point2[1],point2[0]),
                     color=(255,0,0),thickness=3)
        cv2.imshow("trajectory", image_rgb)
        cv2.waitKey()
        # plt.waitforbuttonpress()
        image_counter += 1
