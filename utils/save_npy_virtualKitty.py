import numpy as np
import argparse
import sys
sys.path.append('./')
from datasets.dataset import VirtualKitty

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_images', '--path_images', type=str, default="", required=True, help='Path to model output')
    args = parser.parse_args()
    args_parsed['path_images'] = args.path_images.split(",")
    print("Args parsed ", args_parsed)

if __name__ == "__main__":
    global args_parsed
    # parse_args()
    city_scapes = VirtualKitty(batch_size=1, seg_image=True, depth_image=True, output_classes=16, save_npy=True)
    for batch, targets in city_scapes.load_train(max_percent=1,print_images_load=1):
        pass

    for batch, targets in city_scapes.load_train(train=False,max_percent=1,print_images_load=1):
        pass