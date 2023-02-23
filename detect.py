import argparse
import torch
import matplotlib.pyplot as plt
from UNet.UNed_model import FCN
import os
import numpy as np
import cv2 as cv
from datasets.dataset import VirtualKitty

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', '--source', type=str, default="", required=True, help='Path to dataset images')
    # parser.add_argument('-output', '--output', type=str, default="", required=True, help='Path to save inference images')
    parser.add_argument('-device', '--device', type=int, default=0, required=True, help='Device GPU to execute')
    parser.add_argument('-model', '--model', type=str, default="", required=True, help='Path to model')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=1, required=True, help='Batch size to train')
    args = parser.parse_args()
    args_parsed['source'] = args.source
    # args_parsed['output'] = args.output
    args_parsed['device'] = args.device
    args_parsed['model'] = args.model
    args_parsed['batch_size'] = args.batch_size
    print("Args parsed ", args_parsed)

if __name__ == "__main__":
    global args_parsed
    args_parsed = dict()
    parse_args()
    in_channels = 3
    out_channels = 1
    seg_image = False
    depth_image = False
    if out_channels == 1:
        depth_image = True
    elif out_channels == 15:
        seg_image = True
    elif out_channels == 16:
        seg_image = True
        depth_image = True
    model = FCN(in_channels, out_channels)
    model.load_state_dict(torch.load(args_parsed['model']))
    # model.eval()
    print(model.cuda(device=args_parsed['device']))
    if not os.path.exists("./runs/"):
        os.mkdir("./runs/")
    if not os.path.exists("./runs/detect/"):
        os.mkdir("./runs/detect/")
    detect_runs = len(os.listdir("./runs/detect/"))
    os.mkdir("./runs/detect/run"+str(detect_runs+1))
    os.mkdir("./runs/detect/run"+str(detect_runs+1) + "/output/")
    args_parsed['output'] = "./runs/detect/run"+str(detect_runs+1) + os.path.sep
    n = (40 // args_parsed['batch_size']) # Plot about 40 samples
    cm = "gray" # Color Map
    virtual_kitty = VirtualKitty(args_parsed['source'], args_parsed['batch_size'])
    batch_it = virtual_kitty.load_test()
    # for i in range(n):
        # print("batch nexxt ", next(batch_it).shape)
    i = 0
    print(f"SAMPLE #{i}")
    batch_has_next = True
    while batch_has_next:
        try:
            batch = next(batch_it)
            # ####### Plot images input to debug errors
            # print("Haceindo cosas de batches", len(batch))
            # for image_batch_index in range(len(batch)):
            #     plt.subplot(1,1,1)
            #     # print("image", batch[image_batch])
            #     image = np.moveaxis(batch[image_batch_index], 0, 2)

            #     plt.imshow(image)
            #     # plt.imshow(image_rgb)
            #     plt.show()

            ###################### end debug images ###############################
        # for batch in next(batch_it):
            batch = torch.tensor(batch)
            # Copy to GPU
            batch = batch.float().cuda()
            # targets = targets_one_hot.float().cuda()
            # Forward Propagation
            with torch.no_grad():
                outputs = model(batch)
            # Plot Outputs
            for j in range(args_parsed['batch_size']):
                # Plot these results
                # fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(32, 32))
                image = np.zeros((192, 624,15), dtype=np.int8)
                image_depth = np.zeros((192,624), dtype=np.float32)
                if seg_image:
                    for channel in range(15):
                        predictions = outputs[j,channel].cpu()
                        mapped_predictions = predictions > 0.75
                        image[:,:,channel] = mapped_predictions
                        # plt.imshow(image[:,:,channel])
                        # plt.waitforbuttonpress()
                    image_rgb = virtual_kitty.convert_channels_toRGB(image)
                    # print("Image rgb channels", image_rgb.shape)
                    # OpenCV needs BGR
                    image_BGR = cv.cvtColor(np.float32(image_rgb), cv.COLOR_RGB2BGR)
                # print(np.min(np.array(outputs[j,15].cpu())), np.max(np.array(outputs[j,15].cpu())), "shape", outputs[j,15].cpu().shape)
                if depth_image:
                    image_depth_norm = virtual_kitty.norm(np.array(outputs[j,-1].cpu()))
                    # print("image_depth min", np.min(image_depth_norm), "max", np.max(image_depth_norm))
                    image_depth = virtual_kitty.convert_range_image(image_depth_norm)
                # print("image_depth min", np.min(image_depth), "max", np.max(image_depth))
                # plt.imshow(image_depth, cmap='gray')
                # plt.waitforbuttonpress()
                if seg_image:
                    cv.imwrite(args_parsed['output'] + "output" + os.path.sep + "SEG_SAMPLE_" + str(i) + "_BATCH_SAMPLE_" + str(j) + ".jpg", image_BGR)
                if depth_image:
                    cv.imwrite(args_parsed['output'] + "output" + os.path.sep + "DEPTH_SAMPLE_" + str(i) + "_BATCH_SAMPLE_" + str(j) + ".jpg", image_depth)
                #plt.show()
            # Clear Cache
            del batch
            # del targets
            del outputs
            torch.cuda.empty_cache()
            i += 1
            # break # Only process one batch for every 'i'
        except StopIteration as e:
            batch_has_next = False
            print("There are not more batches")
            continue
