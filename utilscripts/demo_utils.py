## add script for code used in the demo 
## slight variation wrt the one used for trn/tst models

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# import torch.models as models
from torchvision import models
import timm  # for xception model
# from torchvision import models
from genericpath import exists
import os
import mediapipe as mp
import cv2
# import numpy as np
import glob
import re
import numpy as np

# # mediapipe face detection model
# face_detection_model = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# def sort_paths(path):
#     # given a path to a folder, sort the files in the folder by the number after 'hand_occlusion' or 'obj_occlusion' in the folder
#     path = sorted(path, key=lambda x: int(re.search(r'\d+', x).group()))
#     return path


# def get_file_name(path):
#     return os.path.basename(path)

# def create_subfolders(path):
#     # create the subfolder for the demo
#     os.makedirs(path + '/hand_occlusion')
#     os.makedirs(path + '/obj_occlusion')

# # function to extract all the subfolders in a directory
# def extract_subfolders(directory):
#     # return [f.path for f in os.scandir(directory) if f.is_dir()]
#     subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
#     # return subfolders following the order of the subfolders in the directory (not sorted)
#     # follows the alphabetical order of the subfolders in the directory
#     return subfolders

# # function to extract all the files in a directory (following the order of the files in the directory)
# def extract_files(directory):
#     videos = [f.path for f in os.scandir(directory) if f.is_file()]
#     return videos
#     # return [f.path for f in os.scandir(directory) if f.is_file()]


# def face_detection(frame, face_detection_model):
#     # convert frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # detect faces in the frame
#     results = face_detection_model.process(frame_rgb)
#     return results

# def extract_face_from_frame(frame, face_bbox):
#     # extract face from frame
#     x, y, w, h = face_bbox
#     face = frame[y:y+h, x:x+w]
#     return face

# # def enlarge_bbox(bbox_coordinates): # enlarge bbox coordinates by 30%
# #   # resulting bbox coordinates must be bigger than the  original bbox coordinates by 1.3 times
# #     x, y, w, h = bbox_coordinates
# #     x = max(0, x - int(0.15 * w))
# #     y = max(0, y - int(0.15 * h))
# #     w = w + int(0.3 * w)
# #     h = h + int(0.3 * h)
# #     return (x, y, w, h)


# def extract_faces_from_videos(vid_paths, out_paths):
#     for vid, out in zip(vid_paths, out_paths):
#         print('extracting faces from video:', vid)  # print video name
#         print('saving faces to:', out)  # print output path
#         # Path to video file
#         vidObj = cv2.VideoCapture(vid)
#         # Used as counter variable
#         count = 0
#         # checks whether frames were extracted
#         success = 1


#         while success and count < 390: # until extracted 390 frames
#             # vidObj object calls read
#             # function extract frames
#             success, image = vidObj.read()
#             # Saves the frames with frame-count
#             # cv2.imwrite(out_path + "/frame%d.jpg" % count, image)
#             results = face_detection(image, face_detection_model)
#             if results.detections:
#                 # print(len(results.detections))
#                 for detection in results.detections:
#                     bboxC = detection.location_data.relative_bounding_box
#                     ih, iw, _ = image.shape
#                     # x, y, w, h computed from the relative bounding box coordinates (0.0 to 1.0)
#                     # x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#                     # bboxC.xmin * iw -> x coordinate of the top-left corner of the bbox (xmin * image_width)
#                     # bboxC.ymin * ih -> y coordinate of the top-left corner of the bbox
#                     # bboxC.width * iw -> width of the bbox
#                     # bboxC.height * ih -> height of the bbox
#                     x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#                     # bboxC = detection.location_data.relative_bounding_box
#                     # ih, iw, _ = image.shape
#                     # multiply bboxC coordinates by the image width and height to get the bbox coordinates in pixels
#                     # print("x:", x)
#                     # print("y:", y)
#                     # print("w:", w)
#                     # print("h:", h)
#                     # face = extract_face_from_frame(image, (x, y, w, h))
#                     # cv2.imwrite(out + "/frame_mediapipe.jpg" , face)
#                     # --------------------------------------------------------------------- #
#                     # enlarge face bbox by 30% (bbox size * 1.3)
#                     x = max(0, x - int(0.15 * w)) # max(0, x - 0.15 * w) -> x coordinate of the top-left corner of the bbox
#                     y = max(0, y - int(0.15 * h)) # max(0, y - 0.15 * h) -> y coordinate of the top-left corner of the bbox
#                     # the 0.15 * w and 0.15 * h are used to go 15% to the left and 15% to the top of the bbox
#                     w = min(iw, w + int(0.3 * w)) # min(iw, w + 0.3 * w) -> width of the bbox
#                     # w * 1.3 = w + 0.3 * w
#                     # iw = image width (in pixels) is the maximum width of the image
#                     # take the min because the width of the bbox cannot be bigger than the width of the image
#                     # the min function is used to avoid the bbox to go out of the image
#                     # remember that the bbox is a rectangle that contains the face in the image
#                     h = min(ih, h + int(0.3 * h)) # min(ih, h + 0.3 * h) -> height of the bbox
#                     # 574x574
#                     # --------------------------------------------------------------------- #

#                     # # enlarge face bbox by 80% while keeping the aspect ratio
#                     # scale_factor = 1.5 # enlarge by 50%
#                     # new_w = int(w * scale_factor)
#                     # new_h = int(h * scale_factor)
#                     # x = max(0, x - (new_w - w) // 2)
#                     # y = max(0, y - (new_h - h) // 2)
#                     # w = min(iw - x, new_w)
#                     # h = min(ih - y, new_h)
#                     # # 574 x 574
#                     # --------------------------------------------------------------------- #
#                     face = extract_face_from_frame(image, (x, y, w, h))

#                     # save face
#                     cv2.imwrite(out + "/frame%d.jpg" % count, face)
#                     print(f"saving face from frame{count}")
#                     count += 1
#         print("done")
#         vidObj.release()
#         # cv2.destroyAllWindows()

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


import os
import shutil

# check if real and fake frames have the same frames names
def check_frames(frames):
    real_frames = frames['original_obj_occ']
    fake_frames = frames['ghost_obj_occ']

    return set(real_frames) == set(fake_frames)

def organize_frames(dataset_path, save_path):
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # print("Found file:", file) # i.e. frame0.jpg
                # print("In directory:", root) # user_faces/fake/hand_occlusion
                # breakpoint()

                # fix the / in the root path
                root = root.replace('\\', '/')
                print("Fixed root path:", root) # user_faces/fake/hand_occlusion
                # breakpoint()

                save_frame_path = os.path.join(save_path, root.split('/')[-1], root.split('/')[-2])
                # print("Frames path:", save_frame_path) # preprocessed_faces/fake/hand_occlusion
                if not os.path.exists(save_frame_path):
                    os.makedirs(save_frame_path, exist_ok=True)

                # breakpoint()

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
                print("root:", root)
                # ----------------------------------------------------------------------------------------------- #

                # Move the frames to the respective directories following the structure in the hardcoded frames

                if 'hand_occlusion' in root:
                    if file in occ_frames['hand_occlusion']:
                        # print(f"File {file} is in the hardcoded frames for hand occlusion.")
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(occ_save_frame_path, file)
                        print(f"Moving {src_file} to {dst_file}")
                        # os.rename(src_file, dst_file)
                        shutil.move(src_file, dst_file)
                    elif file in no_occ_frames:
                        # print(f"File {file} is in the hardcoded frames for no hand occlusion.")
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(no_occ_save_frame_path, file)
                        print(f"Moving {src_file} to {dst_file}")
                        # os.rename(src_file, dst_file)
                        shutil.move(src_file, dst_file)
                    else:
                        # print(f"File {file} does not match any hardcoded frames for hand occlusion, skipping.")
                        continue
                elif 'obj_occlusion' in root:
                    if file in occ_frames['obj_occlusion']:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(occ_save_frame_path, file)
                        # print(f"Moving {src_file} to {dst_file}")
                        shutil.move(src_file, dst_file)
                    elif file in no_occ_frames:
                        # print(f"File {file} is in the hardcoded frames for no object occlusion.")
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(no_occ_save_frame_path, file)
                        # print(f"Moving {src_file} to {dst_file}")
                        shutil.move(src_file, dst_file)
                    else:
                        # print(f"File {file} does not match any hardcoded frames for object occlusion, skipping.")
                        continue
                else:
                    print(f"Directory {root} does not match any hardcoded frames, skipping.")
# ----------------------------------------------------------------------------------------------- #

def check_num_frames(path):
    # Iterate over all directories and report only those with exactly 100 frames
    for root, dirs, files in os.walk(path):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
        if len(image_files) == 100:
            print(f"Directory {root} contains exactly 100 frames.")

# ----------------------------------------------------------------------------------------------- #


def get_pretrained_path(model_name, trn_strategy, dataset, models_path):
    # Check if the model is valid
    if model_name not in ['mnetv2', 'effnetb4', 'xception']: #, 'icpr2020', 'neurips2023']:
        print(f"Model {model_name} is not supported")
        exit()
    # Check if the dataset is valid
    if dataset not in ['fows_occ', 'fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']:
        print(f"Dataset {dataset} is not valid")
        exit()

    # Get the model path based on the arguments
    model_info = model_name + '_' + dataset + ('_FT' if trn_strategy=='ft' else '_TL')
    # # '_TL' if tl else
    print("Model info: ", model_info)

    if trn_strategy.lower() == 'tl':
        models_folder = models_path+'/TL/'
    else:
        models_folder = models_path+'/FT/' # to be updated to 'model_weights/FT/' if FT is used
    pretrained_model_path = ''

    # Check if the model path exists
    if not os.path.exists(models_folder):
        print(f"Model path {models_folder} does not exist")
        exit()

    # navigate all model folders subdirectories and check if the model_info is in any of them
    found = False
    for root, dirs, files in os.walk(models_path):
        for dir in dirs:
            # print(f"Checking directory: {dir}")
            if dir.lower() == model_name.lower():  # check if the directory name is the same as the model_info
                # find the .pth file in the directory
                print(f"Found directory: {dir} with same name as {model_name}")
                model_dir = os.path.join(root, dir)
                for sub_root, sub_dirs, sub_files in os.walk(model_dir):
                    for file in sub_files:
                        if file.endswith('.pth'):
                            found = True
                            pretrained_model_path = os.path.join(sub_root, file)
                            print(f"Pretrained model path: {pretrained_model_path}")
                            break
                break # uncomment this if you want to stop searching after finding the first directory

    if not found:
        print(f"NO model {model_name} found in {model_name}")
        exit()

    print(f"Pretrained model path: {pretrained_model_path}")

    return pretrained_model_path

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
def get_test_transf():
    # Transformations for testing data
    test_transform = transforms.Compose([
        transforms.Resize((256,256)), 
        # BICUBIC is used for EfficientNetB4 -> check the documentation
        # BICUBIC vs BILINEAR -> https://discuss.pytorch.org/t/what-is-the-difference-between-bilinear-and-bicubic-interpolation/20920 
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return test_transform

# ------------------------------------------------------------------------------------------- #
def model_forward_pass(imgs_tensor, model, device): #, fake_imgs_tensor):
    batch_size = 32
    model.eval()
    all_probs = []
    # fake_all_probs = []
    # num_batches = int(np.ceil(len(imgs_tensor) / batch_size))
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(imgs_tensor), batch_size)):
            # print(f"Batch {batch_idx+1}/{num_batches}")
            batch_imgs = imgs_tensor[i:i+batch_size].to(device)
            outputs = model(batch_imgs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten() # Shape: (batch_size,) 
            # .cpu().numpy() converts the tensor to a numpy array
            # .flatten() converts the array to a 1D array
            all_probs.extend(probs)

            # # compute the probabilities for the fake images
            # fake_batch_imgs = fake_imgs_tensor[i:i+batch_size].to(device)
            # fake_outputs = model(fake_batch_imgs)
            # fake_probs_batch = torch.sigmoid(fake_outputs).cpu().numpy().flatten()
            # fake_all_probs.extend(fake_probs_batch)
    
    return all_probs #, fake_all_probs

# ------------------------------------------------------------------------------------------- #
def load_model_from_path(model_name, trn_strategy, dataset, model_weights_path):
    # if model_name == 'mnetv2':
    pretrained_model_path = 'model_path_not_found'
    model_str = f"{model_name}_{dataset}_{trn_strategy}"
    # if 'mnetv2' in model_name.lower():
    if model_name == 'mnetv2':
        print("Loading MobileNetV2 model")
        # model = mobilenet_v2(pretrained=True)
        model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2')
        model_name = 'MobileNetV2' # add the model name to the model object

        if trn_strategy == 'tl':
            # ---------------------------- #
            # --- TRANSFER LEARNING!!! --- #
            # ---------------------------- #
            print("loading pretrained TL model")
            # # Freeze all layers
            # for param in model.parameters():
            #     param.requires_grad = False
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

            # if dataset == 'gotcha_no_occ':
            #     # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
            #     pretrained_model_path = get_model_path(model_weights_path)
            #     print("model saved in:", pretrained_model_path)
            #     # with torch.serialization.safe_globals([FocalLoss]): # Use context manager
            #     best_ckpt = torch.load(pretrained_model_path, map_location = "cpu") #, weights_only=False)
            #     model.load_state_dict(best_ckpt['model'])

            # elif dataset in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
            #     # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
            #     pretrained_model_path = get_model_path(model_weights_path)
            #     #get_pretrained_path(model_name, dataset, tl, ft)
            #     print("model saved in:", pretrained_model_path)
            #     # with torch.serialization.safe_globals([FocalLoss]): # Use context manager
            #     model.load_state_dict(torch.load(pretrained_model_path))

            pretrained_model_path = get_model_path(model_weights_path, model_str)
            print("model saved in:", pretrained_model_path)
            best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
            # breakpoint()
            if 'model' in best_ckpt.keys():
              model.load_state_dict(best_ckpt['model'])
            else:
              model.load_state_dict(best_ckpt)
            # else:
            #     print("no pretrained model found")
            #     exit()

            print("model saved in:", pretrained_model_path)
            # print("model loaded!")

        elif trn_strategy == 'ft':
        # else:
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            print("loading pretrained FT model")
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1)

            # if dataset in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
            # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
            pretrained_model_path = get_model_path(model_weights_path, model_str)
            print("model saved in:", pretrained_model_path)
            #get_pretrained_path(model_name, dataset, tl, ft)
            # with torch.serialization.safe_globals([FocalLoss]): # Use context manager

            best_ckpt = torch.load(pretrained_model_path, map_location='cpu', weights_only= False)
            model.load_state_dict(best_ckpt['model'])
            # model.load_state_dict(torch.load(pretrained_model_path))
        else:
            print("no pretrained model found")
            exit()

        # # model.to(device)
        # print("Model loaded!")
        # # print(model)
        # print("model_name: ", model_name)
        # print("tst_dataset: ", dataset)
        # print("trn_strategy: ", trn_strategy)
        # print("model_path: ", pretrained_model_path)

    # elif 'effnetb4' in model_name.lower():
    elif model_name == 'effnetb4':
        print("Loading EfficientNetB4 model")
        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        model_name = 'EfficientNet_B4' # add the model name to the model object

        # load the pre-trained model for testing (load model weights)
        if trn_strategy == 'tl':
            # ---------------------------- #
            # --- TRANSFER LEARNING!!! --- #
            # ---------------------------- #
            print("loading pretrained TL model")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Replace the classifier layer
            # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face

            if dataset == 'gotcha_no_occ':
                # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
                pretrained_model_path = get_model_path(model_weights_path, model_str)
                print("model saved in:", pretrained_model_path)
                #get_pretrained_path(model_name, dataset, tl, ft)
                # model.load_state_dict(torch.load(pretrained_model_path))
                # with torch.serialization.safe_globals([FocalLoss]): # Use context manager
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
                model.load_state_dict(best_ckpt['model'])
            elif dataset == 'gotcha_occ':
                pretrained_model_path = get_model_path(model_weights_path, model_str)
                print("model saved in:", pretrained_model_path)
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
                model.load_state_dict(best_ckpt['model'])

            elif dataset in ['fows_occ', 'fows_no_occ']:
                # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
                pretrained_model_path = get_model_path(model_weights_path, model_str)
                print("model saved in:", pretrained_model_path)
                #get_pretrained_path(model_name, dataset, tl, ft)
                # print("model saved in:", pretrained_model_path)
                # with torch.serialization.safe_globals([FocalLoss]): # Use context manager

                model.load_state_dict(pretrained_model_path)
            else:
                print("no pretrained model found")
                exit()

        elif trn_strategy == 'ft':
        # else:
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            print("loading pretrained FT model")
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            # if args.dataset == 'thesis_occ':
            # if dataset in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
            # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
            pretrained_model_path = get_model_path(model_weights_path, model_str)
            print("model saved in:", pretrained_model_path)

            # with torch.serialization.safe_globals([FocalLoss]): # Use context manager
            best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
            model.load_state_dict(best_ckpt['model'])
        else:
            print("no pretrained model found")
            exit()


        # model.to(device)
        # print("Model loaded!")
        # print("model_name: ", model_name)
        # print("tst_dataset: ", dataset)
        # print("trn_strategy: ", trn_strategy)
        # print("model_path: ", pretrained_model_path)

    elif model_name == 'xception':
        print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        # pip install timm
        # import timm
        model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
        model_name = 'XceptionNet' # add the model name to the model object
        # load the pre-trained model for testing (load model weights)
        # pretrained_model_path = args.save_model_path
        if trn_strategy == 'tl':
            # ---------------------------- #
            # --- TRANSFER LEARNING!!! --- #
            # ---------------------------- #
            print("loading pretrained model")

            if dataset in ['gotcha_occ', 'gotcha_no_occ']:
                # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
                pretrained_model_path = get_model_path(model_weights_path, model_str)
                print("model saved in:", pretrained_model_path)
                # get_pretrained_path(model_name, dataset, tl, ft)
                # model.load_state_dict(torch.load(pretrained_model_path))
                # with torch.serialization.safe_globals([BCEWithLogitsLoss]): # Use context manager
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
                model.load_state_dict(best_ckpt['model'])
            elif dataset in ['fows_occ', 'fws_no_occ']:
                # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
                pretrained_model_path = get_model_path(model_weights_path, model_str)
                print("model saved in:", pretrained_model_path)
                #get_pretrained_path(model_name, dataset, tl, ft)
                # print("model saved in:", pretrained_model_path)
                # with torch.serialization.safe_globals([FocalLoss]): # Use context manager
                model.load_state_dict(torch.load(pretrained_model_path))

            else:
                print("no pretrained model found")
                exit()

        elif trn_strategy == 'ft':
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            if model_name in ['fows_occ, fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']:
                print("loading pretrained FT model")
                # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
                pretrained_model_path = get_model_path(model_weights_path, model_str)
                print("model saved in:", pretrained_model_path)

                # ------------------------------------------ #
                print("model saved in:", pretrained_model_path)
                # with torch.serialization.safe_globals([FocalLoss]): # Use context manager
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
        else:
            print("no pretrained model found")
            exit()

        # model.to(device)
        # print("model loaded!")
        # print("model_name: ", model_name)
        # print("tst_dataset: ", dataset)
        # print("trn_strategy: ", trn_strategy)
        # print("model_path: ", pretrained_model_path)

    else:
        print("Model not supported")
        exit()

    return model, pretrained_model_path