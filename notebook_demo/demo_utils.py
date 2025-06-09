## add script for code used in the demo 
## slight variation wrt the one used for trn/tst models
import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# import torch.models as models
from torchvision import models
import timm  # for xception model
from genericpath import exists
import cv2
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #
# hardcoded frames extraction for the occ and no_occ frames
# Hand_occ_1

occ_frames = {
    'hand_occlusion': [
            "frame200.jpg",
            "frame201.jpg",
            "frame202.jpg",
            "frame203.jpg",
            "frame204.jpg",
            "frame205.jpg",
            "frame206.jpg",
            "frame207.jpg",
            "frame208.jpg",
            "frame209.jpg",
            "frame210.jpg",
            "frame211.jpg",
            "frame212.jpg",
            "frame213.jpg",
            "frame214.jpg",
            "frame215.jpg",
            "frame216.jpg",
            "frame217.jpg",
            "frame218.jpg",
            "frame219.jpg",
            "frame220.jpg",
            "frame221.jpg",
            "frame222.jpg",
            "frame223.jpg",
            "frame224.jpg",
            "frame225.jpg",
            "frame226.jpg",
            "frame227.jpg",
            "frame228.jpg",
            "frame229.jpg",
            "frame230.jpg",
            "frame231.jpg",
            "frame232.jpg",
            "frame233.jpg",
            "frame234.jpg",
            "frame235.jpg",
            "frame236.jpg",
            "frame237.jpg",
            "frame238.jpg",
            "frame239.jpg",
            "frame240.jpg",
            "frame241.jpg",
            "frame242.jpg",
            "frame243.jpg",
            "frame244.jpg",
            "frame245.jpg",
            "frame246.jpg",
            "frame247.jpg",
            "frame248.jpg",
            "frame249.jpg",
            "frame250.jpg",
            "frame251.jpg",
            "frame252.jpg",
            "frame253.jpg",
            "frame254.jpg",
            "frame255.jpg",
            "frame256.jpg",
            "frame257.jpg",
            "frame258.jpg",
            "frame259.jpg",
            "frame260.jpg",
            "frame261.jpg",
            "frame262.jpg",
            "frame263.jpg",
            "frame264.jpg",
            "frame265.jpg",
            "frame266.jpg",
            "frame267.jpg",
            "frame268.jpg",
            "frame269.jpg",
            "frame270.jpg",
            "frame271.jpg",
            "frame272.jpg",
            "frame273.jpg",
            "frame274.jpg",
            "frame275.jpg",
            "frame276.jpg",
            "frame277.jpg",
            "frame278.jpg",
            "frame279.jpg",
            "frame280.jpg",
            "frame281.jpg",
            "frame282.jpg",
            "frame283.jpg",
            "frame284.jpg",
            "frame285.jpg",
            "frame286.jpg",
            "frame287.jpg",
            "frame288.jpg",
            "frame289.jpg",
            "frame290.jpg",
            "frame291.jpg",
            "frame292.jpg",
            "frame293.jpg",
            "frame294.jpg",
            "frame295.jpg",
            "frame296.jpg",
            "frame297.jpg",
            "frame298.jpg",
            "frame299.jpg" ],

    'obj_occlusion': [
        "frame190.jpg",
        "frame191.jpg",
        "frame192.jpg",
        "frame193.jpg",
        "frame194.jpg",
        "frame195.jpg",
        "frame196.jpg",
        "frame197.jpg",
        "frame198.jpg",
        "frame199.jpg",
        "frame200.jpg",
        "frame201.jpg",
        "frame202.jpg",
        "frame203.jpg",
        "frame204.jpg",
        "frame205.jpg",
        "frame206.jpg",
        "frame207.jpg",
        "frame208.jpg",
        "frame209.jpg",
        "frame210.jpg",
        "frame211.jpg",
        "frame212.jpg",
        "frame213.jpg",
        "frame214.jpg",
        "frame215.jpg",
        "frame216.jpg",
        "frame217.jpg",
        "frame218.jpg",
        "frame219.jpg",
        "frame220.jpg",
        "frame221.jpg",
        "frame222.jpg",
        "frame223.jpg",
        "frame224.jpg",
        "frame225.jpg",
        "frame226.jpg",
        "frame227.jpg",
        "frame228.jpg",
        "frame229.jpg",
        "frame230.jpg",
        "frame231.jpg",
        "frame232.jpg",
        "frame233.jpg",
        "frame234.jpg",
        "frame235.jpg",
        "frame236.jpg",
        "frame237.jpg",
        "frame238.jpg",
        "frame239.jpg",
        "frame240.jpg",
        "frame241.jpg",
        "frame242.jpg",
        "frame243.jpg",
        "frame244.jpg",
        "frame245.jpg",
        "frame246.jpg",
        "frame247.jpg",
        "frame248.jpg",
        "frame249.jpg",
        "frame250.jpg",
        "frame251.jpg",
        "frame252.jpg",
        "frame253.jpg",
        "frame254.jpg",
        "frame255.jpg",
        "frame256.jpg",
        "frame257.jpg",
        "frame258.jpg",
        "frame259.jpg",
        "frame260.jpg",
        "frame261.jpg",
        "frame262.jpg",
        "frame263.jpg",
        "frame264.jpg",
        "frame265.jpg",
        "frame266.jpg",
        "frame267.jpg",
        "frame268.jpg",
        "frame269.jpg",
        "frame270.jpg",
        "frame271.jpg",
        "frame272.jpg",
        "frame273.jpg",
        "frame274.jpg",
        "frame275.jpg",
        "frame276.jpg",
        "frame277.jpg",
        "frame278.jpg",
        "frame279.jpg",
        "frame280.jpg",
        "frame281.jpg",
        "frame282.jpg",
        "frame283.jpg",
        "frame284.jpg",
        "frame285.jpg",
        "frame286.jpg",
        "frame287.jpg",
        "frame288.jpg",
        "frame289.jpg"
    ],

}

no_occ_frames = [
       "frame0.jpg",
        "frame1.jpg",
        "frame10.jpg",
        "frame11.jpg",
        "frame12.jpg",
        "frame13.jpg",
        "frame14.jpg",
        "frame15.jpg",
        "frame16.jpg",
        "frame17.jpg",
        "frame18.jpg",
        "frame19.jpg",
        "frame2.jpg",
        "frame20.jpg",
        "frame21.jpg",
        "frame22.jpg",
        "frame23.jpg",
        "frame24.jpg",
        "frame25.jpg",
        "frame26.jpg",
        "frame27.jpg",
        "frame28.jpg",
        "frame29.jpg",
        "frame3.jpg",
        "frame30.jpg",
        "frame31.jpg",
        "frame32.jpg",
        "frame33.jpg",
        "frame34.jpg",
        "frame35.jpg",
        "frame36.jpg",
        "frame37.jpg",
        "frame38.jpg",
        "frame39.jpg",
        "frame4.jpg",
        "frame40.jpg",
        "frame41.jpg",
        "frame42.jpg",
        "frame43.jpg",
        "frame44.jpg",
        "frame45.jpg",
        "frame46.jpg",
        "frame47.jpg",
        "frame48.jpg",
        "frame49.jpg",
        "frame5.jpg",
        "frame50.jpg",
        "frame51.jpg",
        "frame52.jpg",
        "frame53.jpg",
        "frame54.jpg",
        "frame55.jpg",
        "frame56.jpg",
        "frame57.jpg",
        "frame58.jpg",
        "frame59.jpg",
        "frame6.jpg",
        "frame60.jpg",
        "frame61.jpg",
        "frame62.jpg",
        "frame63.jpg",
        "frame64.jpg",
        "frame65.jpg",
        "frame66.jpg",
        "frame67.jpg",
        "frame68.jpg",
        "frame69.jpg",
        "frame7.jpg",
        "frame70.jpg",
        "frame71.jpg",
        "frame72.jpg",
        "frame73.jpg",
        "frame74.jpg",
        "frame75.jpg",
        "frame76.jpg",
        "frame77.jpg",
        "frame78.jpg",
        "frame79.jpg",
        "frame8.jpg",
        "frame80.jpg",
        "frame81.jpg",
        "frame82.jpg",
        "frame83.jpg",
        "frame84.jpg",
        "frame85.jpg",
        "frame86.jpg",
        "frame87.jpg",
        "frame88.jpg",
        "frame89.jpg",
        "frame9.jpg",
        "frame90.jpg",
        "frame91.jpg",
        "frame92.jpg",
        "frame93.jpg",
        "frame94.jpg",
        "frame95.jpg",
        "frame96.jpg",
        "frame97.jpg",
        "frame98.jpg",
        "frame99.jpg"
    ]


def organize_frames(dataset_path, save_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # fix the / in the root path
                root = root.replace('\\', '/')
                save_frame_path = os.path.join(save_path, root.split('/')[-1], root.split('/')[-2])
                # print("Frames path:", save_frame_path) # preprocessed_faces/fake/hand_occlusion
                if not os.path.exists(save_frame_path):
                    os.makedirs(save_frame_path, exist_ok=True)
                # root = path to frames dir
                occ_save_frame_path = os.path.join(save_frame_path, 'occ')
                no_occ_save_frame_path = os.path.join(save_frame_path, 'no_occ')
                # print("Occ frames path:", occ_save_frame_path)
                # print("No occ frames path:", no_occ_save_frame_path)
                # breakpoint()
                if not os.path.exists(occ_save_frame_path):
                    os.makedirs(occ_save_frame_path, exist_ok=True)
                if not os.path.exists(no_occ_save_frame_path):
                    os.makedirs(no_occ_save_frame_path, exist_ok=True) # exist_ok=True will not raise an error if the directory already exists
                # breakpoint()
                # print("root:", root)
                # ----------------------------------------------------------------------------------------------- #

                # Move the frames to the respective directories following the structure in the hardcoded frames

                if 'hand_occlusion' in root:
                    if file in occ_frames['hand_occlusion']:
                        # print(f"File {file} is in the hardcoded frames for hand occlusion.")
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(occ_save_frame_path, file)
                        # print(f"Moving {src_file} to {dst_file}")
                        # os.rename(src_file, dst_file)
                        # shutil.move(src_file, dst_file)
                        shutil.copy(src_file, dst_file)
                        
                    elif file in no_occ_frames:
                        # print(f"File {file} is in the hardcoded frames for no hand occlusion.")
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(no_occ_save_frame_path, file)
                        # print(f"Moving {src_file} to {dst_file}")
                        # os.rename(src_file, dst_file)
                        # shutil.move(src_file, dst_file)
                        shutil.copy(src_file, dst_file)
                    else:
                        # print(f"File {file} does not match any hardcoded frames for hand occlusion, skipping.")
                        continue
                elif 'obj_occlusion' in root:
                    if file in occ_frames['obj_occlusion']:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(occ_save_frame_path, file)
                        # print(f"Moving {src_file} to {dst_file}")
                        # shutil.move(src_file, dst_file)
                        shutil.copy(src_file, dst_file)

                    elif file in no_occ_frames:
                        # print(f"File {file} is in the hardcoded frames for no object occlusion.")
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(no_occ_save_frame_path, file)
                        # print(f"Moving {src_file} to {dst_file}")
                        # shutil.move(src_file, dst_file) # move frame from src to dst
                        shutil.copy(src_file, dst_file) # copy frame from src to dst
                    else:
                        # print(f"File {file} does not match any hardcoded frames for object occlusion, skipping.")
                        continue
                else:
                    # print(f"Directory {root} does not match any hardcoded frames, skipping.")
                    continue
# ----------------------------------------------------------------------------------------------- #

def check_num_frames(path):
    # Iterate over all directories and report only those with exactly 100 frames
    for root, dirs, files in os.walk(path):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
        if len(image_files) == 100:
            print(f"Directory {root} contains exactly 100 frames.")

# ------------------------------------------------------------------------------------------- #
def get_test_transf():
    # Transformations for testing data
    test_transform = transforms.Compose([
        transforms.Resize((256,256)), 
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return test_transform

# ------------------------------------------------------------------------------------------- #
def model_forward_pass(imgs_tensor, model, device):
    batch_size = 32
    # model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(imgs_tensor), batch_size)):
            # print(f"Batch {batch_idx+1}/{num_batches}")
            batch_imgs = imgs_tensor[i:i+batch_size].to(device)
            outputs = model(batch_imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten() # Shape: (batch_size,) 
            all_probs.extend(probs)
    
    return all_probs 

# ----------------------------------------------------------------------------------------------- #
def plot_prob_graph(original_prob, fake_prob, model_str, frames_type):
    plt.plot(original_prob, label='Original Predictions', alpha=0.8)
    plt.plot(fake_prob, label='Fake Predictions', alpha=0.8)
    plt.axhline(0.5, color='k', linestyle='--', label='Threshold = 0.5')  # Add threshold line at 0.5
    plt.title(f'{model_str} - Original {frames_type} Predictions VS Fake {frames_type} Predictions')
    plt.xlabel('Frames')
    plt.ylabel('Model score')
    plt.legend()
    plt.tight_layout()
    plt.show()  # Show the plot

# ------------------------------------------------------------------------------------------- #
def get_model_path(model_weights_path, model_str):
  model_path = ''
  for root, dirs, files in os.walk(model_weights_path):
      for file in files:
        if file.endswith('.pth') and model_str in file:
          print(os.path.join(root, file))
          model_path = os.path.join(root, file)
  print(model_path)
  return model_path

# ------------------------------------------------------------------------------------------- #
def load_model_from_path(model_name, pretrained_model_path):

    # pretrained_model_path = get_model_path()

    if model_name == 'mnetv2':
        print("Loading MobileNetV2 model")
        model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2')
        model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)

    elif model_name == 'effnetb4':
        print("Loading EfficientNetB4 model")
        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)

    elif model_name == 'xception':
        print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        # pip install timm
        # import timm
        model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(torch.load(pretrained_model_path))
    else:
        print("Model not supported")
        sys.exit()

    return model, pretrained_model_path
