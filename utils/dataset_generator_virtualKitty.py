import argparse
import random
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_images', '--dataset_images', type=str, default="", required=True, help='Path to dataset images')
    args = parser.parse_args()
    args_parsed['data_path'] = args.dataset_images
    print("Args parsed ", args_parsed)


def create_dataset_file(images_path):
    absolutes_images_path = []
    for path in Path(images_path).rglob('*.jpg'):
        absolutes_images_path.append(str(path.absolute()))
    
    random.shuffle(absolutes_images_path)
    print("absolutes", absolutes_images_path)
    train_images = absolutes_images_path[:int(len(absolutes_images_path)*0.8)]
    test_images = absolutes_images_path[int(len(absolutes_images_path)*0.8):]
    with open('./datasets/train.txt', 'w') as f:
        f.write('\n'.join(train_images)) 
    with open('./datasets/test.txt', 'w') as f:
        f.write('\n'.join(test_images)) 

if __name__ == "__main__":
    global args_parsed
    args_parsed = dict()
    parse_args()
    create_dataset_file(args_parsed['data_path'])