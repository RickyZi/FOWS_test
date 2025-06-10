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

# ----------------------------------------------------------------------------------------------- #
# TODO: update code to load models (use args)
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
    model_info = model_name + '_' + dataset + ('_FT' if trn_strategy=='FT' else '_TL')
    # # '_TL' if TL else
    print("Model info: ", model_info)

    if trn_strategy.lower() == 'TL':
        models_folder = models_path+'/TL/'
    else:
        models_folder = models_path+'/FT/' # to be updated to 'model_weights/FT/' if FT is used
    pretrained_model_path = ''

    # Check if the model path exists
    if not os.path.exists(models_folder):
        print(f"Model path {models_folder} does not exist")
        sys.exit()

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
                break # uncomment this if you want to stop searching aFTer finding the first directory

    if not found:
        print(f"NO model {model_name} found in {model_name}")
        sys.exit()

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
def load_model_from_path(args, model_weights_path):
    # if model_name == 'mnetv2':
    # pretrained_model_path = 'model_path_not_found'
    model_str = f"{args.model}_{args.train_dataset}_{'TL' if args.tl else 'FT'}"
    # trn_strategy = model_str.split('_')[-1]
    # model_name = model_str.split('_')[0]
    # dataset = model_str.split('_')[1] + '_' + model_str.split('_')[2]

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = get_model_path(model_weights_path, model_str)
    # pretrained_model_path = get_pretrained_path(model_name, trn_strategy, dataset, model_weights_path)
    if pretrained_model_path:
        print("model path: ", pretrained_model_path)
        print("loading the model...")
    else:
        print("no pretrained model found")
        exit()


    # if 'mnetv2' in model_name.lower():
    if args.model == 'mnetv2':
        print("Loading MobileNetV2 model")
        # model = mobilenet_v2(pretrained=True)
        model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2')
        # model_name = 'MobileNetV2' # add the model name to the model object
        # Replace the classifier layer
        model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
        
        # pretrained_model_path = get_model_path(model_weights_path, model_str)
        # print("model saved in:", pretrained_model_path)
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)
        
        # model.to(device)

    # elif 'effnetb4' in model_name.lower():
    elif args.model == 'effnetb4':
        print("Loading EfficientNetB4 model")
        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        # model_name = 'EfficientNet_B4' # add the model name to the model object
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face

        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        # breakpoint()
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(best_ckpt)
        


    elif args.model == 'xception':
        print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        # pip install timm
        # import timm
        model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
        # model_name = 'XceptionNet' # add the model name to the model object
        # load the pre-trained model for testing (load model weights)
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu", weights_only=False)
        if 'model' in best_ckpt.keys():
            model.load_state_dict(best_ckpt['model'])
        else:
            model.load_state_dict(torch.load(pretrained_model_path))


    else:
        print("Model not supported")
        sys.exit()

    return model, pretrained_model_path

# --------------------------------------------------------- #

def get_backbone(args):
    if args.model == 'mnetv2': 
        print("Loading MobileNetV2 model")

        if args.tl:
            print("Loading MobileNetV2 pre-trained model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

            # check the model gradient
            # check_model_gradient(model)
            # ()
            model_name = 'MobileNetV2_TL' # add the model name to the model object
        
        elif args.ft:
            print("Fine-Tuning MobileNetV2 pre-trained model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            print("Transfer learning - Freezing all layers except the classifier")
            # # Freeze all layers
            # for param in model.parameters():
            #     param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
            if args.resume:
                # if args.tl and args.resume:
                print("Resume training from checkpoint for MobileNetV2")
                print("checkpoint path:", args.resume)
                pretrained_model_path = args.resume
                # model_name = "XceptionNet_TL"
                checkpoint = torch.load(args.resume, map_location = "cpu")
                # model = timm.create_model('xception', pretrained=True)
                model.load_state_dict(checkpoint['model'])
            # check the model gradient
            # check_model_gradient(model)
            # ()
            model_name = 'MobileNetV2_FT' # add the model name to the model object

        else:
            # print("problem in loading the model")
            # exit()
            print("Loading MobileNetV2 pre-trained model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

            # check the model gradient
            # check_model_gradient(model)
            # ()
            model_name = 'MobileNetV2' # add the model name to the model object

        # model.to(device)
        # print("Model loaded!")

        # # print(model)
        # # ()
        # # exit()

    elif args.model == 'effnetb4':
        print("Loading EfficientNetB4 model")
        # model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
        # pip install efficientnet_pytorch
        # run this command to install the efficientnet model
        # model = EfficientNet.from_pretrained('efficientnet-b4')

        if args.tl: # and args.resume != '':
            print("Loading EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html
            print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real or swap face
            # last layer of EfficientNetB4 is a Linear layer (classifier) with 1000 outputs (for ImageNet) -> change it to 1 output

            model_name = 'EfficientNet_B4' # add the model name to the model object
            model.to(device)
            # check the model gradient
            # check_model_gradient(model)
            
        elif args.ft: 
            print("Fine-Tuning the EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real or swap face
            model_name = 'EfficientNet_B4_FT'
            
            # # model.to(device)
            # # # check the model gradient
            # # check_model_gradient(model)
            # # # ()

            
        else: 
            # print("problem in loading the model")
            # exit()
            print("Loading EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            # model.to(device)
            model_name = 'EfficientNet_B4' # add the model name to the model object
        # load the pre-trained model for testing (load model weights)
        # pretrained_model_path = args.save_model_path 
        # print("model saved in:", pretrained_model_path)
        # model.load_state_dict(torch.load(pretrained_model_path)) 
        # model.to(device)
        print("Model loaded!")

        # print(model)
        # exit()
        
    elif args.model == 'xception':
        # print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        # pip install timm
        # import timm

        # if args.tl and args.resume:
        #     print("Resume training from checkpoint for XceptionNet")
        #     print("checkpoint path:", args.resume)
        #     model_name = "XceptionNet_TL"
        #     checkpoint = torch.load(args.resume, map_location = "cpu")
        #     model = timm.create_model('xception', pretrained=True)
        #     model.load_state_dict(checkpoint['model'])
        #     model.to(device)
        #     print('Model loaded!')


        if args.tl:
            print("Loading XceptionNet pre-trained model - Transfer learning - Freezing all layers except the classifier")
            # print("Transfer learning - Freezing all layers except the classifier")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            # print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers except the classifier
            for param in model.parameters():
                param.requires_grad = False

            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

            # Ensure the classifier layer is trainable
            for param in model.fc.parameters():
                param.requires_grad = True
           
            # # check the model gradient
            # check_model_gradient(model)

            if args.resume:
                # if args.tl and args.resume:
                print("Resume training from checkpoint for XceptionNet")
                print("checkpoint path:", args.resume)
                pretrained_model_path = args.resume
                model_name = "XceptionNet_TL"
                checkpoint = torch.load(args.resume, map_location = "cpu")
                # model = timm.create_model('xception', pretrained=True)
                model.load_state_dict(checkpoint['model'])
                # model.to(device)
                # print('Model loaded!')

            # print('Model loaded!')
            # model.to(device)

        elif args.ft: 
            print("Fine Tuning XceptionNet pre-trained model")
            # print("Transfer learning - Freezing all layers except the classifier")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            # print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers except the classifier
            # for param in model.parameters():
            #     param.requires_grad = False

            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

            # # Ensure the classifier layer is trainable
            # for param in model.fc.parameters():
            #     param.requires_grad = True
            if args.resume:
                # if args.tl and args.resume:
                print("Resume training from checkpoint for XceptionNet")
                print("checkpoint path:", args.resume)
                pretrained_model_path = args.resume
                model_name = "XceptionNet_TL"
                checkpoint = torch.load(args.resume, map_location = "cpu")
                # model = timm.create_model('xception', pretrained=True)
                model.load_state_dict(checkpoint['model'])
            # check the model gradient
            # check_model_gradient(model)

        else: 
            # print("problem in loading the model")
            # exit()
            print("Loading XceptionNet pre-trained model")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            # print("Transfer learning - Freezing all layers except the classifier")
            # # Freeze all layers except the classifier
            # for param in model.parameters():
            #     param.requires_grad = False

            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

            # Ensure the classifier layer is trainable
            for param in model.fc.parameters():
                param.requires_grad = True
           
            # check the model gradient
            # check_model_gradient(model)
        # else:
        #     # retraining the model with one output class 
        #     model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real or swap face
        #     model_name = 'XceptionNet' # add the model name to the model object

        

        # model.to(device)
        # print("Model loaded!")

        print(model)
        # exit()
        # check the model gradient
        # check_model_gradient(model)


    else:
        print("Model not supported")
        exit()

    return model, model_name
