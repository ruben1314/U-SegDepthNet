import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import matplotlib.image as mtpi
import time
import torch
from torchvision import transforms
import sys
sys.path.append('./monodepth2/')
import networks
from utils import download_model_if_doesnt_exist

model_name = "mono_640x192"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval();

def inference(image, image_name, original_height, original_width, images_path):
    start_time = time.time()
    with torch.no_grad():
        features = encoder(image)
        outputs = depth_decoder(features)
    final_depth_time = time.time() - start_time
    disp = outputs[("disp", 0)]
    ## Plotting
    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)

    # plt.figure(figsize=(10, 10))
    # plt.subplot(211)
    # plt.imshow(input_image)
    # plt.title("Input", fontsize=22)
    # plt.axis('off')

    plt.subplot(212)
    plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
    import cv2 as cv
    # color_map_image = cv.applyColorMap(disp_resized_np, cv.COLORMAP_MAGMA)
    # cv.imwrite("./assets/out/"+image_name + ".jpg", color_map_image)
    mtpi.imsave("./assets/out/"+image_name + ".jpg", disp_resized_np, vmax=vmax, cmap='magma')
    # plt.savefig("./assets/out/"+image_name + ".jpg", cv.COLORMAP_MAGMA)
    # plt.title("Disparity prediction", fontsize=22)
    return disp_resized_np,vmax, final_depth_time

def predict_images(images_path):
    # print("images path dpeth", images_path)
    for image_path in images_path:
        # image_path = os.path.join(images_path, image)
        image_path = image_path.replace('\n','')
        name,ext = os.path.splitext(os.path.basename(image_path))
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        img_result, vmax, final_depth_time = inference(input_image_pytorch, name, original_height, original_width, input_image)
        yield img_result, vmax, final_depth_time




