import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from PIL import Image
import argparse
from torch.utils.data import Dataset
from utilscripts.get_trn_tst_model import *
from utilscripts.customDataset import FaceImagesDataset
from utilscripts.get_trn_tst_model import get_pretrained_model
# import utilscripts.fornet as fornet
# from utilscripts.fornet import *
# from utilscripts.papers_data_augmentations import *

# https://github.com/jacobgil/pytorch-grad-cam/tree/master
# pip install pip install grad-cam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM


def compute_gradacm(model, test_dataloader, device, model_name, exp_results_path, cam_method, num_layers= 1, gotcha = False):
    model.eval()
    # correct = 0
    # total = 0

    # Define the target layer for Grad-CAM
    if 'mnetv2' in model_name:
        if num_layers == 1:
            target_layers = model.features[-1]
        elif num_layers == 2:
            target_layers = [model.features[-2], model.features[-1]]
        elif num_layers == 3:
            target_layers = [model.features[-3], model.features[-2], model.features[-1]]
        else:
            raise ValueError(f"Unsupported number of target layers: {num_layers}")
        # target_layer = model.features[-1]
    
    elif 'effnetb4' in model_name:
        if num_layers == 1:
            # target_layers = [model.features[-1]]
            target_layers = [model.features[-2]]  # Last Conv2dNormActivation layer as in effnetb4_dfdc
            # target_layers = [model.features[-2], model.features[-1]]
        elif num_layers == 2:
            target_layers = [model.features[-2], model.features[-1]]  # Last two layers
        elif num_layers == 3:
            target_layers = [model.features[-3], model.features[-2], model.features[-1]]  # Last three layers
        else:
            raise ValueError(f"Unsupported number of target layers: {num_layers}")

    elif 'xception' in model_name:
        # target_layers = [model.backbone.conv4]
        if num_layers == 1:
            target_layers = [model.conv4]
        elif num_layers == 2:
            target_layers = [model.conv3, model.conv4]
        elif num_layers == 3:
            target_layers = [model.conv2, model.conv3, model.conv4]
        else:
            raise ValueError(f"Unsupported number of target layers: {num_layers}")

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Initialize the CAM-method
    if cam_method == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif cam_method == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif cam_method == 'eigencam':
        cam = EigenCAM(model=model, target_layers=target_layers)
    elif cam_method == 'scorecam':
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unsupported CAM method: {cam_method}")

    results = {
        'image_path': [],
        'label': [],
        'prediction': [],
    }

    for i, (images, labels, image_paths) in enumerate(test_dataloader):
        print(f"Batch {i}") # print the batch number -> 1 image per batch (batch_size = 1)
        print(images.size()) # (batch_size, 3, 224, 224)
        print(labels.size()) # (batch_size, 1)

        print("len(images): ", len(images))
        print("len(labels): ", len(labels))
        print("len(image_paths): ", len(image_paths))
        # Enable gradients
        images.requires_grad = True

        # Forward pass
        output = model(images.to(device))

        # Get the predicted class
        _, predicted = torch.max(output.data, 1)
        print("img_path: ", image_paths)
        print("labels: ", labels)
        print("predicted: ", predicted)
        
        # if gotcha: 
        #     # if user_id == '42': 
        #         if 'original' in image_paths[0]:
        #             user_id = image_paths[0].split('/')[-4]
        #             if user_id == '42':
        #                 frame_id = image_paths[0].split('/')[-1].split('.')[0]
        #                 challenge_id = image_paths[0].split('/')[-2]
        #                 algo_id = image_paths[0].split('/')[-3]
        #                 swap_id = None
        #             else: continue
        #         else:
        #             user_id = image_paths[0].split('/')[-5]
        #             if user_id == '42':
        #                 frame_id = image_paths[0].split('/')[-1].split('.')[0]
        #                 swap_id = image_paths[0].split('/')[-2]
        #                 challenge_id = image_paths[0].split('/')[-3]
        #                 algo_id = image_paths[0].split('/')[-4]
        #             else: 
        #                 continue
        # else: 
        # get info to save the image from image_paths[0]
        frame_id = image_paths[0].split('/')[-1].split('.')[0]
        challenge_id = image_paths[0].split('/')[-2]
        algo_id = image_paths[0].split('/')[-3]
        user_id = image_paths[0].split('/')[-4]
        # swap_id = None

        
        # ---------------------------------------------------------------------------------------------------------- #
        grayscale_cam = cam(input_tensor=images.to(device), targets=[BinaryClassifierOutputTarget(labels.item())])[0]
        # as discussed here: https://github.com/jacobgil/pytorch-grad-cam/issues/325 
        # BinaryClassifierOutputTarget -> if the net has only one output with a sigmoid
        # ---------------------------------------------------------------------------------------------------------- #
        # Read the original image
        rgb_img = cv2.imread(image_paths[0])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.float32(rgb_img) / 255
        rgb_img_resized = cv2.resize(rgb_img, (224, 224))

        # print rgb_img_resized.shape
        print("rgb_img_resized", rgb_img_resized.shape) # (224, 224, 3)
        # print the greyscale_cam heatmap shape
        print("grayscale_cam", grayscale_cam.shape) # (224, 224)
        # 

        # Overlay the heatmap on the original image
        cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
        cam_subfolders_path = f"{exp_results_path}/{user_id}/{algo_id}/"
        os.makedirs(cam_subfolders_path, exist_ok=True)
       

        # if gotcha: 
        #     if swap_id: 
        #         # print(swap_id)
        #         cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{swap_id}_{frame_id}.png"
        #         results['image_path'].append(f"{user_id}_{algo_id}_{challenge_id}_{swap_id}_{frame_id}")
        #         results['label'].append(labels.cpu().numpy())
        #         results['prediction'].append(predicted.cpu().numpy())
        #         # print(cam_image_path)
        #         # 
        #     else:
        #         cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
        #         results['image_path'].append(f"{user_id}_{algo_id}_{challenge_id}_{frame_id}")
        #         results['label'].append(labels.cpu().numpy())
        #         results['prediction'].append(predicted.cpu().numpy())
        # else: 
        # print("no swap_id")
        cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
        results['image_path'].append(f"{user_id}_{algo_id}_{challenge_id}_{frame_id}")
        results['label'].append(labels.cpu().numpy())
        results['prediction'].append(predicted.cpu().numpy())
            
        # save gradcam
        cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM image saved to: {cam_image_path}")
        # 


    print("len(results['image_path'])", len(results['image_path']))
    print("len(results['label'])", len(results['label']))
    print("len(results['prediction'])", len(results['prediction']))
    print(f"Batch {i}") # should be 14 
    # breakpoint()

    print("Grad-CAM testing completed!")
    results_table = pd.DataFrame(results)
    print("results: \n", results_table)

    # save the results table in a log file called results.log in the same folder as the gradcam result
    results_table.to_csv(f"{exp_results_path}/results.log", index=False)


    
def get_args_parse():
    parser = argparse.ArgumentParser(description='Grad-CAM testing')
    parser.add_argument('--model', type=str, default='mnetv2', help='Model name')
    parser.add_argument('--dataset', type=str, default='thesis_occ', help='Dataset name')
    parser.add_argument('--tl', action='store_true', help='Use transfer learning model')
    parser.add_argument('--ft', action='store_true', help='Use fine-tuned model')
    # parser.add_argument('--tags', type=str, default='BinaryClassifierOutputTarget', help='Target type')
    parser.add_argument('--method', type=str, default='gradcam++', choices=['gradcam', 'gradcam++', 'eigencam', 'scorecam'], help='CAM method')
    parser.add_argument('--num-layers', type=int, default=1, help = 'Set num of target layers for CAM analysis')

    return parser

def main():

    # Parse the arguments
    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    print(args)
    gotcha = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("model:", args.model)
    print("ft: ", args.ft)
    print("tl: ", args.tl)
    print("num_layers: ", args.num_layers)
    
    # pretrained_model_path = get_pretrained_model_path(args, model)

    # print("model_path: ", pretrained_model_path)
    
    # load the model to the GPU
    # load the model
    # if args.model == 'icpr2020':
    #     net_name = "EfficientNetB4"
    #     net_class = getattr(fornet, net_name)
    #     model: FeatureExtractor = net_class().to(device)
    #     model_state = torch.load(pretrained_model_path, map_location = "cpu")
    #     incomp_keys = model.load_state_dict(model_state['net'], strict=True)
    #     print(incomp_keys)
    #     print(model)
    #     print('Model loaded!')
    #     args.dataset = 'milan_occ' if args.dataset == 'thesis_occ' else 'milan_no_occ' # replace the dataset name with the one used in the training
    #     # args.data_aug = 'milan'
    #     model_name = args.model
    # elif args.model == 'neurips2023':
    #     net_name = "DFB_xceptionNet"
    #     # @TODO: check how to load the DFB model and test it 
    # else:
    # print("model saved in:", pretrained_model_path)
    # model_name = args.model
    # # model.load_state_dict(torch.load(pretrained_model_path))
    # best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    # model.load_state_dict(best_ckpt['model']) 
    # model.to(device)
    # print(model)
    # print("Model loaded!")

    # print("args.dataset: ", args.dataset)
    # breakpoint()

    model, model_name, pretrained_model_path = get_pretrained_model(args, device)
    model.to(device)
    print("Model loaded!")
    print("model_path: ", pretrained_model_path)
    print(model)

    # # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # if args.dataset == 'milan_occ' or args.dataset == 'milan_no_occ': 
    #     # add the Milan Augmentation
    #     milan_transforms = milan_test_transf()
    # else: 
    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # select the test dataset
    if args.dataset == 'fows_occ':
        test_dataset = FaceImagesDataset('/home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_thesis/thesis_occ/', test_transform)
            # '/media/data/rz_dataset/users_face_occlusion/testing/', test_transform)
    elif args.dataset == 'fows_no_occ':
        test_dataset = FaceImagesDataset('/home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_thesis/thesis_no_occ/', test_transform)
    # elif args.dataset == 'gotcha_occ':
    #     test_dataset = FaceImagesDataset('/home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/gotcha_occ_testing', test_transform)
    #     gotcha = True
    # elif args.dataset == 'gotcha_no_occ':
    #     test_dataset = FaceImagesDataset('/home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/gotcha_no_occ_testing', test_transform)
    #     gotcha = True
    # elif args.dataset == 'milan_occ':
    #     test_dataset = FaceImagesDataset('/home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_milan/occlusion_testing/', milan_transforms, use_albu=True)
    # elif args.dataset == 'milan_no_occ':
    #     test_dataset = FaceImagesDataset('/home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_milan/no_occlusion_testing/', milan_transforms, use_albu=True)
    else: 
        raise ValueError(f"Unsupported dataset name: {args.dataset}")
    # test_dataset = FaceImagesDataset(directory='/content/drive/MyDrive/WORK/test_gradcam/rand_imgs/thesis_occ/', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # exp_results_path = f'/content/drive/MyDrive/WORK/test_gradcam/gradcam_output/test_{model_name}_gradcam_BinaryClassifierOutputTarget_gradcam/'
    if args.tl:
        model_name = args.model + '_TL'
    else: 
        model_name = args.model + '_FT'

    if args.num_layers == 1:
        # exp_results_path = f'/home/rz/rz-test/bceWLL_test/gradcam_output/{model_name}_{args.dataset}_{args.method}/'#_{args.tags}/'
        exp_results_path = f'/home/rz/rz-test/bceWLL_test/facenet_cross_models_gradcam/{model_name}_{args.dataset}_{args.method}/'#_{args.tags}/'
    else: 
        # exp_results_path = f'/home/rz/rz-test/bceWLL_test/gradcam_output/{model_name}_{args.dataset}_{args.method}_{args.num_layers}_layers/'
        exp_results_path = f'/home/rz/rz-test/bceWLL_test/facenet_cross_models_gradcam/{model_name}_{args.dataset}_{args.method}_{args.num_layers}_layers/'
    os.makedirs(exp_results_path, exist_ok=True)
    
    compute_gradacm(model, test_dataloader, device, args.model, exp_results_path, args.method, args.num_layers, gotcha) #, use_gradcam_plus_plus=False)

    print("done")
if __name__ == '__main__':
    main()

    #python test_gradcam.py --model "effnetb4_dfdc" --dataset "milan_occ" --tl --method "gradcam++"