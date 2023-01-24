import os
import re
from UNet.UNed_model import FCN
import torch
import time
import numpy as np
import argparse
from datasets.dataset import VirtualKitty

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', '--source', type=str, default="", required=True, help='Path to dataset images')
    parser.add_argument('-output', '--output', type=str, default="", required=True, help='Path to model output')
    parser.add_argument('-device', '--device', type=int, default=0, required=True, help='Device GPU to execute')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=1, required=True, help='Batch size to train')
    parser.add_argument('-model', '--model', type=str, default="", required=False, help='Path to model')
    parser.add_argument('-dataset_reduction', '--dataset_reduction', type=float, default=1.0, required=False, help='Percent to reduct dataset')
    args = parser.parse_args()
    args_parsed['source'] = args.source
    args_parsed['output'] = args.output
    args_parsed['device'] = args.device
    args_parsed['batch_size'] = args.batch_size
    args_parsed['model'] = args.model
    args_parsed['dataset_reduction'] = args.dataset_reduction
    print("Args parsed ", args_parsed)

if __name__ == "__main__":
    global args_parsed
    args_parsed = dict()
    parse_args()
    in_channels = 3
    out_channels = 15
    model = FCN(in_channels, out_channels)
    if args_parsed['model'] != "":
        model.load_state_dict(torch.load(args_parsed['model']))
    print(model.cuda(device=args_parsed["device"]))

    # Hyperparameters
    epochs = 100
    learning_rate = 1.0e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Setting up a global loss function
    criterion = FocalTverskyLoss

    train_avg_losses = []
    test_avg_losses = []
    # Additional metrics
    avg_iou_scores = []
    avg_dice_scores = []

    TRAINING_START_TIME = time.time()
    virtual_kitty = VirtualKitty(args_parsed["source"], args_parsed["batch_size"])

    for epoch in range(epochs):
        print("############################### EPOCH ", epoch, " ###############################")
        EPOCH_START_TIME = time.time()
        
        epoch_train_losses = []
        epoch_test_losses = []
        epoch_iou_scores = [] # Computed only on test set
        epoch_dice_scores = [] # Computed only on test set
        
        for batch, targets in virtual_kitty.load_train(max_percent=args_parsed['dataset_reduction']):
            # Convert samples to one-hot form
            batch = torch.tensor(batch)
            targets = torch.tensor(targets)
            targets_one_hot = targets > 0
            # Copy to GPU
            batch = batch.float().cuda()
            targets = targets_one_hot.float().cuda()
            # Train
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, targets) 
            epoch_train_losses.append(float(loss))
            loss.backward()
            optimizer.step()
            # Clear Cache
            del batch
            del targets
            del outputs
            torch.cuda.empty_cache()

        for batch, targets in virtual_kitty.load_train(train=False,max_percent=args_parsed['dataset_reduction']):
            # Convert samples to one-hot form
            batch = torch.tensor(batch)
            targets = torch.tensor(targets)
            targets_one_hot = targets > 0
            # Copy to GPU
            batch = batch.float().cuda()
            targets = targets_one_hot.float().cuda()
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
                
        EPOCH_END_TIME = time.time()
        
        train_avg_loss = np.average(epoch_train_losses)
        test_avg_loss = np.average(epoch_test_losses)
        iou_avg_score = np.average(epoch_iou_scores)
        dice_avg_score = np.average(epoch_dice_scores)
        
        train_avg_losses.append(train_avg_loss)
        test_avg_losses.append(test_avg_loss)
        avg_iou_scores.append(iou_avg_score)
        avg_dice_scores.append(dice_avg_score)

        training_time_stamp = int(EPOCH_END_TIME - TRAINING_START_TIME)
        epoch_time_taken = int(EPOCH_END_TIME - EPOCH_START_TIME)
        ep = f"{epoch+1}".zfill(2)
        print(f"[{ep}/{epochs}]  TRAIN:{train_avg_loss:.5f}  TEST:{test_avg_loss:.5f}  AVG_DICE_SCORE:{dice_avg_score:.5f}  AVG_IOU_SCORE:{iou_avg_score:.5f}  TOOK:{epoch_time_taken}s (t:{training_time_stamp}s)")
        
        # Save the model every 10 epochs
        if ((epoch+1) % 10) == 0:
            torch.save(model.state_dict(), args_parsed["output"] + os.path.sep + f"basicUNET_epoch{(epoch+1)}.torch")

    for epoch_model in os.listdir(args_parsed["output"]):
        if re.search ("epoch", epoch_model):
            os.remove(args_parsed["output"] + epoch_model)
    torch.save(model.state_dict(),args_parsed["output"] + os.path.sep + "UNet.torch")