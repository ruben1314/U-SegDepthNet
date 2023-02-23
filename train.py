import os
import re
from UNet.UNed_model import FCN
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
# from pytorchsummary import summary
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datasets.dataset import VirtualKitty
import cv2 as cv
from torch.utils.tensorboard import SummaryWriter


def IOU(preds, targets, smooth=0.001):
    preds = preds.view(-1)
    targets = targets.view(-1)
    # Intersection is equivalent to True Positive count
    # Union is the mutually inclusive area of all labels & predictions 
    intersection = (preds & targets).float().sum()
    union = (preds | targets).float().sum()
    # Compute Score
    IoU = (intersection + smooth) / (union + smooth)
    return IoU

# FocalTversky
def FocalTverskyLoss(predictions, targets, alpha=0.7, beta=0.3, gamma=4/3, smooth=0.001):
    TP = (predictions * targets).sum()    
    FP = ((1 - targets) * predictions).sum()
    FN = (targets * (1 - predictions)).sum()
    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)  
    FocalTversky = (1 - Tversky) ** gamma
    return FocalTversky

def accuracy(predictions, targets, batch_size):
    predictions = torch.moveaxis(predictions,1,3)
    targets = torch.moveaxis(targets,1,3)
    accuracy = []
    for i in range(batch_size):
        tensor_shape = predictions.shape
        # print("tensor shape", tensor_shape, predictions[i].shape,targets[i].shape)
        hits = predictions[i]==targets[i]
        print("hits ", hits.shape)
        trues = np.ones(tensor_shape[3], dtype=bool)
        # print("trues", trues)
        print(np.all(hits==trues, axis=-1))
        hits_trues = np.where(np.all(hits==trues, axis=-1))
        # print("hits ", hits_trues)
        hits_sum = len(hits_trues[0])
        accuracy.append((hits_sum*100) / (tensor_shape[2]*tensor_shape[1]))
        # print("accuracy percent", (hits_sum*100) / (tensor_shape[2]*tensor_shape[1]))
    print("average accuracy ", np.average(accuracy))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output', '--output', type=str, default="", required=True, help='Path to model output')
    parser.add_argument('-device', '--device', type=int, default=0, required=True, help='Device GPU to execute')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=1, required=True, help='Batch size to train')
    parser.add_argument('-model', '--model', type=str, default="", required=False, help='Path to model')
    parser.add_argument('-dataset_reduction', '--dataset_reduction', type=float, default=1.0, required=False, help='Percent to reduct dataset')
    parser.add_argument('-print_test', '--print_test', type=int, default=5, required=False, help='Epochs to calculate average test loss')
    parser.add_argument('-print_images_load', '--print_images_load', type=int, default=100, required=False, help='Images load to print percent')
    parser.add_argument('-epochs', '--epochs', type=int, default=100, required=False, help='Epochs to train. Default:100')
    args = parser.parse_args()
    args_parsed['output'] = args.output
    args_parsed['device'] = args.device
    args_parsed['batch_size'] = args.batch_size
    args_parsed['model'] = args.model
    args_parsed['dataset_reduction'] = args.dataset_reduction
    args_parsed['print_test'] = args.print_test
    args_parsed['print_images_load'] = args.print_images_load
    args_parsed['epochs'] = args.epochs
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
    if args_parsed['model'] != "":
        model.load_state_dict(torch.load(args_parsed['model']))
    print(model.cuda(device=args_parsed["device"]))
    for name, parameter in model.named_parameters():
        print("Parameter", name, parameter.numel())
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Numero de parametros totales", pytorch_total_params)
    # Hyperparameters
    # epochs = 100
    learning_rate = 1.0e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Setting up a global loss function
    criterion = FocalTverskyLoss
    mse_loss = MSELoss()
    train_avg_losses = []
    test_avg_losses = []
    # Additional metrics
    avg_iou_scores = []
    avg_dice_scores = []
    len_runs_training = 0 
    try:
        len_runs_training = len(os.listdir("./runs/train/"))
    except:
        print("No runs")
    os.makedirs("./runs/train/run" + str(len_runs_training+1))
    TRAINING_START_TIME = time.time()
    virtual_kitty = VirtualKitty(batch_size=args_parsed["batch_size"], seg_image=seg_image, depth_image=depth_image, output_classes=out_channels)
    writer = SummaryWriter()
    for epoch in range(args_parsed['epochs']):
        print("############################### EPOCH ", epoch, " ###############################")
        EPOCH_START_TIME = time.time()
        
        epoch_train_losses = []
        epoch_test_losses = []
        epoch_iou_scores = [] # Computed only on test set
        epoch_dice_scores = [] # Computed only on test set
        epoch_iou_train_scores = []
        losses_seg_train_epoch = []
        losses_depth_train_epoch = []
        i = 0
        for batch, targets in virtual_kitty.load_train(max_percent=args_parsed['dataset_reduction'],print_images_load=args_parsed['print_images_load']):
            # ####### Plot images input to debug errors
            # print("Haceindo cosas de batches", len(batch))
            # for image_batch_index in range(len(batch)):
            #     plt.subplot(6,3,1)
            #     # print("image", batch[image_batch])
            #     image = np.moveaxis(batch[image_batch_index], 0, 2)

            #     plt.imshow(image)
            #     for channel_index in range(15):
            #         # print("channel", channel.shape)
            #         plt.subplot(6,3,channel_index+2)
            #         image_seg = np.moveaxis(targets[image_batch_index], 0, 2)
            #         plt.imshow(image_seg[:,:,channel_index])
            #     plt.subplot(6,3,17)
            #     # plt.imshow(image_rgb)
            #     plt.show()

            ###################### end debug images ###############################
            # Convert samples to one-hot form
            batch = torch.tensor(batch)
            targets = torch.tensor(targets)
            if seg_image:
                targets_one_hot = targets[:,:15] > 0
            # Copy to GPU
            batch = batch.float().cuda(device=args_parsed["device"])
            if seg_image:
                targets[:,:15] = targets_one_hot.float()
            targets = targets.cuda(device=args_parsed["device"])
            # Train
            optimizer.zero_grad()
            outputs = model(batch)
            with torch.no_grad():
                    if seg_image:
                        predictions = outputs.cpu() > 0.75
                        iou_score = float(IOU(predictions, targets_one_hot))
                        # accuracy(predictions[:,:15], targets[:,:15].cpu(),args_parsed['batch_size'])
                        epoch_iou_train_scores.append(iou_score)
                        # epoch_dice_scores.append(dice_score)
            # if True:
            #     for j in range(args_parsed['batch_size']):
            #         # Plot these results
            #         # fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(32, 32))
            #         image = np.zeros((192, 624,15), dtype=np.int8)
            #         for channel in range(15):
            #             predictions = outputs[j,channel].cpu()
            #             mapped_predictions = predictions > 0.75
            #             image[:,:,channel] = mapped_predictions
                        # plt.imshow(image[:,:,channel])
                        # plt.waitforbuttonpress()
                    # image_rgb = virtual_kitty.convert_channels_toRGB(image)
                    # print("Image rgb channels", image_rgb.shape)
                    # OpenCV needs BGR
                    # image_BGR = cv.cvtColor(np.float32(image_rgb), cv.COLOR_RGB2BGR)
                    # plt.imshow(image_rgb)
                    # plt.waitforbuttonpress()
                    # print("Creando ","./runs/train/run" + str(len_runs_training+1) + "/epoca" + str(epoch))
                    # os.makedirs("./runs/train/run" + str(len_runs_training+1) + "/epoca" + str(epoch), exist_ok=True)
                    # print("Ruta imagen", )
                    # cv.imwrite("./runs/train/run" + str(len_runs_training+1) + "/epoca" + str(epoch) + os.path.sep + "SAMPLE_" + str(i) + "_BATCH_SAMPLE_" + str(j) + ".jpg", image_BGR)
            
            loss_seg = 0
            loss_depth = 0
            if seg_image:
                loss_seg = criterion(outputs[:,:15], targets[:,:15]) 
            print("Outputs", outputs.shape,"targets", targets.shape)
            if depth_image:
                loss_depth = mse_loss(outputs[:,-1],targets[:,-1])
            loss = loss_seg + loss_depth
                
            # TODO MSE depth sumar perdidas, 
            # TODO separa funcion activacion para seg y depth seg --> sftmax, depth --> sigmoid
            # TODO hacer solo profundidad
            epoch_train_losses.append(float(loss))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if seg_image:
                    losses_seg_train_epoch.append(loss_seg.cpu())
                if depth_image:
                    losses_depth_train_epoch.append(loss_depth.cpu())
            # Clear Cache
            del batch
            del targets
            del outputs
            torch.cuda.empty_cache()
            i += 1

        if (epoch % args_parsed['print_test']) == 0:
            for batch, targets in virtual_kitty.load_train(train=False,max_percent=args_parsed['dataset_reduction'],print_images_load=args_parsed['print_images_load']):
                # Convert samples to one-hot form
                batch = torch.tensor(batch)
                targets = torch.tensor(targets)
                targets_one_hot = targets > 0
                # Copy to GPU
                batch = batch.float().cuda(device=args_parsed["device"])
                targets = targets_one_hot.float().cuda(device=args_parsed["device"])
                # Forward Propagation
                with torch.no_grad():
                    outputs = model(batch)
                    loss = float(criterion(outputs, targets))
                    predictions = outputs.cpu() > 0.75
                    iou_score = float(IOU(predictions, targets_one_hot))
                    # dice_score = float(DiceScore(predictions, targets_one_hot))
                    epoch_iou_scores.append(iou_score)
                    # epoch_dice_scores.append(dice_score)
                    epoch_test_losses.append(loss)
                # Clear Cache
                del batch
                del targets
                del outputs
                torch.cuda.empty_cache()
            test_avg_loss = np.average(epoch_test_losses)
            iou_avg_score = np.average(epoch_iou_scores)
            # dice_avg_score = np.average(epoch_dice_scores)

            test_avg_losses.append(test_avg_loss)
            avg_iou_scores.append(iou_avg_score)
            # avg_dice_scores.append(dice_avg_score)

        # for n_iter in range(100):
        if (epoch % args_parsed['print_test']) == 0:
            writer.add_scalar('Loss/test',test_avg_loss, epoch)
            writer.add_scalar('Accuracy/test', iou_avg_score, epoch)
        writer.add_scalar('Loss/train_seg', np.average(losses_seg_train_epoch), epoch)
        writer.add_scalar('Loss/train_depth', np.average(losses_depth_train_epoch), epoch)
        writer.add_scalar('Accuracy/train', iou_score, epoch)

                
        EPOCH_END_TIME = time.time()
        
        train_avg_loss = np.average(epoch_train_losses)
        
        train_avg_losses.append(train_avg_loss)
        
        training_time_stamp = int(EPOCH_END_TIME - TRAINING_START_TIME)
        epoch_time_taken = int(EPOCH_END_TIME - EPOCH_START_TIME)
        ep = f"{epoch+1}".zfill(2)
        print(f"[{ep}/{args_parsed['epochs']}]  TOOK:{epoch_time_taken}s (TOTAL:{training_time_stamp}s TRAIN:{train_avg_loss:.5f}",end="")
        if (epoch % args_parsed['print_test'] == 0):
            print(f" TEST:{test_avg_loss:.5f}  AVG_IOU_SCORE:{iou_avg_score:.5f}", end="")
        print(")")
        
        # Save the model every 10 epochs
        if ((epoch+1) % 10) == 0:
            torch.save(model.state_dict(), args_parsed["output"] + os.path.sep + f"basicUNET_epoch{(epoch+1)}.torch")

    for epoch_model in os.listdir(args_parsed["output"]):
        if re.search ("epoch", epoch_model):
            os.remove(args_parsed["output"] + epoch_model)
    torch.save(model.state_dict(),args_parsed["output"] + os.path.sep + "UNet.torch")