 import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
# from torchvision.models import mobilenet_v2
import os
from utils.customDataset import FaceImagesDataset
# from utils.albuDataLoader import FaceImagesAlbu
from utils.train_val_test import *
# from gotcha_trn_val_tst import gotcha_test
# import cv2
# import wandb # for logging results to wandb
import argparse # for command line arguments
# pip install efficientnet_pytorch # need to install this package to use EfficientNet
# from efficientnet_pytorch import EfficientNetdata

from utils.logger import * # import the logger functions
import timm # for using the XceptionNet model (pretrained)
# pip install timm # install the timm package to use the XceptionNet model
# from utils.papers_data_augmentations import *

# import fornet
# from fornet import *
# import yaml
import random
# from train import FocalLoss
# from focalLoss import FocalLoss
from test_utils import *
from utils.get_pretrained_model import *

# -------------------------------------------------------------------------------- #
# test the TL models
# MILAN_FF
# python test.py --model "efficientnetb4_ff" --dataset "milan_occ" --data-aug "milan"  --save-log --tags "EffNetB4_FF_milan_occ_TL" --tl-model
# python test.py --model "efficientnetb4_dfdc" --dataset "milan_occ" --data-aug "milan"  --save-log --tags "EffNetB4_DFDC_milan_occ_TL" --tl-model
# python test.py --model "effnetb4_ff" --dataset "gotcha_occ" --batch-size 128 --tl --tags "EffNetB4_FF_gotcha_occ_TL" --data-aug "milan"


# ------------------------------------------------------------------------------------------- #
# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # Add arguments

    # model parameters
    parser.add_argument('--model', type=str, default='mnetv2', help='Define which trained model to load, e.g. mnetv2_fows_occ')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--thresh', type = float, default = 0.5, help = 'Threshold for binary classification.')
    parser.add_argument('--save-log', action='store_false', help='Save the model output logs')
    # parser.add_argument('--save-model-path', action= "store_false", help='Use the saved pre-trained version of the model')
    parser.add_argument('--tl', action = 'store_true', help='Use the re-trained version of the model (transf learning)')
    parser.add_argument('--ft', action = 'store_true', help='Fine-Tuning the model')
    # parser.add_argument('--gen', action = 'store_true', help='Testing the generalization capabilites of the model')
    # parser.add_argument('--robust', action = 'store_true', help='Testing the robusteness of the model to different data augmentations')
    parser.add_argument('--data-aug', type=str, default='default', help='Data augmentation to use for training and testing') 
    parser.add_argument('--dataset', type=str, default='fows_occ', help='Path to the testing dataset')
    # parser.add_argument('--wandb', action='store_true', help='Use wandb for logging') # if not specified in the command as --wandb the value is set to False
    parser.add_argument('--tags', type=str, default='face-occlusion', help='Info about the model, training setting, dataset, test setting, etc.')
    
    return parser


def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # Overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # Load the new state dict
    model.load_state_dict(model_dict)

def init_seed():
    # --------------------------------------------- #
    # init_seed def taken from DFB code
    # Set the random seed for reproducibility
    random.seed(42)
    # np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print("fixed random seed!")
    # --------------------------------------------- #

def check_model_gradient(model):
    # Check if the gradient is active for the FC layer
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        # if 'fc' in name:
        #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        # else:
        #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")

# ----------------------------------------- #
# main function to train and test the model #
# ----------------------------------------- #
def main(): 
    # Parse the arguments
    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    # initialize the random seed
    init_seed()

    # if args.scratch and args.model == 'mobilenetv2':
    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # ----------------------- #
    # --- BASELINE MODELS --- #
    # ----------------------- #

    # print(args.save_model_path)

    print("args.tl: ", args.tl)
    print("args.ft: ", args.ft)
    print("args.gen: ", args.gen)

    print("thresh: ", args.thresh)
    # 

    print("args.model: ", args.model)
    print("args.dataset: ", args.dataset)
    print("args.data_aug: ", args.data_aug)
    



    
    # # if args.model == 'mnetv2': 
    # if 'mnetv2' in args.model.lower():
    #     print("Loading MobileNetV2 model")
    #     # model = mobilenet_v2(pretrained=True)
    #     model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2') 

    #     model_name = 'MobileNetV2' # add the model name to the model object


    #     if args.tl:
    #         # ---------------------------- #
    #         # --- TRANSFER LEARNING!!! --- #
    #         # ---------------------------- #
    #         print("loading pretrained TL model")
    #         # # Freeze all layers
    #         for param in model.parameters():
    #             param.requires_grad = False
            
    #         # Replace the classifier layer
    #         model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
    #         # if args.dataset == 'thesis_occ':
    #         if 'thesis_occ' in args.model:
    #             # load the pre-trained model for testing (load model weights)
    #             pretrained_model_path = './model/best_focal_loss_model/new_focal_loss_no_resize_rotation_color_jitter.pth'  #args.save_model_path
    #             model.load_state_dict(torch.load(pretrained_model_path)) 
    #         # elif args.dataset == 'thesis_no_occ':
    #         elif 'thesis_no_occ' in args.model:
    #             pretrained_model_path = './model/MobileNetV2_baseline_no_occ/best_model.pth'
    #             model.load_state_dict(torch.load(pretrained_model_path))
    #         # elif args.dataset == 'gotcha_occ': 
    #         elif 'gotcha_occ' in args.model:
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/TL/results_TL/results_TL/MobileNetV2_gotcha_occ_TL/training/model/best_model_1.pth'
    #             model.load_state_dict(torch.load(pretrained_model_path))
    #         # elif args.dataset == 'gotcha_no_occ':
    #         elif 'gotcha_no_occ' in args.model:
    #             pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_TL/MnetV2_gotcha_no_occ_TL/training/MobileNetV2_TL_2025-02-26-16-01-26/best_checkpoint.pth'
    #             # model.load_state_dict(torch.load(pretrained_model_path))
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])

    #         # elif 'originalpos' in args.model.lower():
    #         #     pretrained_model_path = './results/results_TL/MnetV2_OriginalPos_TL/training/MobileNetV2_TL_2025-01-18-17-55-01/best_checkpoint.pth'
    #         #     best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #         #     model.load_state_dict(best_ckpt['model'])

    #         else:
    #             print("no pretrained model found")
    #             exit()

    #         print("model saved in:", pretrained_model_path)
    #         print("model loaded!")

    #     elif args.ft:
    #         # ---------------------- #
    #         # --- FINE TUNING!!! --- #
    #         # ---------------------- #
    #         print("loading pretrained FT model")
    #         # # Freeze all layers
    #         # for param in model.parameters():
    #         #     param.requires_grad = False
            
    #         # Replace the classifier layer
    #         model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
    #         # if args.dataset == 'thesis_occ':
    #         if 'thesis_occ' in args.model:
    #             if args.data_aug == 'new':
    #                 pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/MnetV2_thesis_occ_FT_data_aug_new/training/MobileNetV2_FT_2025-01-25-15-06-44/best_checkpoint.pth'
    #             elif args.data_aug == 'crop':
    #                 print("no CenterCrop data aug model!")
    #                 exit()
    #             else:
    #                 pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/MobileNetV2_thesis_occ_FT/training/MobileNetV2_FT_2024-10-17-15-33-53/best_checkpoint.pth'  #args.save_model_path 
    #             # retrained model
                
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'thesis_no_occ':
    #         elif 'thesis_no_occ' in args.model:
    #             if args.data_aug == 'new':
    #                 pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/MnetV2_thesis_no_occ_FT_data_aug_new/training/MobileNetV2_FT_2025-01-25-15-12-42/best_checkpoint.pth'
    #             elif args.data_aug == 'crop':
    #                 print("no CenterCrop data aug model!")
    #                 exit()
    #             else:
    #                 pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/MobileNetV2_thesis_no_occ_FT/training/MobileNetV2_FT_2024-10-17-15-46-47/best_checkpoint.pth'
    #             # retrained model
                
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'gotcha_occ': 
    #         elif 'gotcha_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/MnetV2_gotcha_occ_FT_data_aug_new/training/MobileNetV2_FT_2025-01-25-15-17-58/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else:
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/MnetV2_gotcha_occ_FT_Adam_opt/training/MobileNetV2_FT_2025-02-17-14-16-56/best_checkpoint.pth'
    #                 #
    #             # retrained model
                
    #             # [best model with AdamW] pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/MobileNetV2_gotcha_occ_FT/training/MobileNetV2_FT_2024-10-21-08-42-05/best_checkpoint.pth
                
    #             # test with Adam
    #             pretrained_model_path = './results/results_FT/MnetV2_gotcha_occ_ADAM_FT/training/MobileNetV2_FT_2025-03-06-08-48-33/best_checkpoint.pth'
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'gotcha_no_occ':
    #         elif 'gotcha_no_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/MnetV2_gotcha_no_occ_FT_data_aug_new/training/MobileNetV2_FT_2025-01-25-16-01-32/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = './results/results_FT/EffNetB4_new_thesis_occ_FT_data_aug_crop/training/EfficientNet_B4_FT_2025-01-24-09-46-19/best_checkpoint.pth'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else: 
                
    #             # [best model AdamW] pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/MnetV2_gotcha_no_occ_FT/training/MobileNetV2_FT_2025-01-21-12-18-06/best_checkpoint.pth'
                
    #             # test with Adam
    #             pretrained_model_path = './results/results_FT/MnetV2_gotcha_no_occ_ADAM_FT/training/MobileNetV2_FT_2025-03-06-09-02-57/best_checkpoint.pth'
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         else:
    #             print("no pretrained model found")
    #             exit()

    #         # print("model saved in:", pretrained_model_path)
    #         # model.load_state_dict(torch.load(pretrained_model_path)) 

    #     else: 
    #         print("using pretrained ImageNet model")
    #         pretrained_model_path = 'MobileNetV2 Pre-trained ImageNet Model'
    #         # 
    #         # Replace the classifier layer
    #         model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
    #         # check_model_gradient(model)
    #         # 

    #     model.to(device)
    #     print("Model loaded!")
    #     # print(model)
    #     print("args.model: ", args.model)
    #     print("args.dataset: ", args.dataset)
    #     print("model_path: ", pretrained_model_path)
    #     # exit()
    #     # 
    #     # print(model)
    #     # check_model_gradient(model)
    #     # breakpoint()


    # # elif args.model == 'effnetb4':
    # elif 'effnetb4' in args.model.lower():
    #     print("Loading EfficientNetB4 model")
    #     # breakpoint()
    #     # pip install efficientnet_pytorch
    #     # run this command to install the efficientnet model
    #     # model = EfficientNet.from_pretrained('efficientnet-b4')

    #     model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
    #     # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html
    #     model_name = 'EfficientNet_B4' # add the model name to the model object

    #     # load the pre-trained model for testing (load model weights)
    #     if args.tl:
    #         # ---------------------------- #
    #         # --- TRANSFER LEARNING!!! --- #
    #         # ---------------------------- #
    #         print("loading pretrained TL model")
    #         # Freeze all layers
    #         for param in model.parameters():
    #             param.requires_grad = False
            
    #         # Replace the classifier layer
    #         # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
    #         model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
            
    #         # if args.dataset == 'thesis_occ':
    #         if 'thesis_occ' in args.model:
    #             # load the pre-trained model for testing (load model weights)
    #             pretrained_model_path = '/home/rz/rz-test/bceWLL_test/model/EfficientNet_B4_baseline_occ/new_best_model.pth'  #args.save_model_path 
    #             print("model saved in:", pretrained_model_path)
    #             model.load_state_dict(torch.load(pretrained_model_path))
                
    #         elif 'thesis_no_occ' in args.model:
    #             pretrained_model_path = '/home/rz/rz-test/bceWLL_test/model/EfficientNet_B4_baseline_no_occ/best_mode__2.pth'
    #             print("model saved in:", pretrained_model_path)
    #             model.load_state_dict(torch.load(pretrained_model_path))
    #             # there are 2 models ...
    #         # elif args.dataset == 'gotcha_occ':
    #         elif 'gotcha_occ' in args.model:
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/TL/results_TL/EffNetB4_gotcha_occ_TL/training/EfficientNet_B4_2024-10-08-09-29-28/best_checkpoint.pth'
    #             print("model saved in:", pretrained_model_path)
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'gotcha_no_occ':
    #         elif 'gotcha_no_occ' in args.model:
    #             pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_TL/EffNetB4_gotcha_no_occ_TL/training/EfficientNet_B4_2025-02-26-16-24-56/best_checkpoint.pth'
    #             #'/media/data/model_exp_results/face_occ_net/TL/results_TL/EffNetB4_gotcha_no_occ_TL/training/EfficientNet_B4_2024-10-08-11-26-30/best_checkpoint.pth'
    #             print("model saved in:", pretrained_model_path)
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         else:
    #             print("no pretrained model found")
    #             exit()

    #         # print("model saved in:", pretrained_model_path)
    #         # model.load_state_dict(torch.load(pretrained_model_path))

    #     elif args.ft:
    #         # ---------------------- #
    #         # --- FINE TUNING!!! --- #
    #         # ---------------------- #
    #         print("loading pretrained FT model")
    #         # Replace the classifier layer
    #         model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    #         # if args.dataset == 'thesis_occ':
    #         if 'thesis_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = './results/results_FT/EffNetB4_thesis_occ_FT_data_aug_new/training/EfficientNet_B4_FT_2025-01-24-16-02-42/best_checkpoint.pth'
    #             #     #'./results/results_FT/EffNetB4_new_thesis_occ_FT_data_aug_new/training/EfficientNet_B4_FT_2025-01-23-18-43-39/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = './results/results_FT/EffNetB4_new_thesis_occ_FT_data_aug_crop/training/EfficientNet_B4_FT_2025-01-24-09-46-19/best_checkpoint.pth'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else: 
    #             # load the pre-trained model for testing (load model weights)
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/EffNetB4_thesis_occ_FT/training/EfficientNet_B4_FT_2024-10-18-11-25-09/best_checkpoint.pth'  #args.save_model_path 
                
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'thesis_no_occ':
    #         elif 'thesis_no_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = './results/results_FT/EffNetB4_thesis_no_occ_FT_data_aug_new/training/EfficientNet_B4_FT_2025-01-24-16-14-57/best_checkpoint.pth'
    #             #     #'./results/results_FT/EffNetB4_new_thesis_no_occ_FT_data_aug_new/training/EfficientNet_B4_FT_2025-01-23-18-58-16/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             #     # pretrained_model_path = './results/results_FT/EffNetB4_new_thesis_no_occ_FT_data_aug_crop/training/EfficientNet_B4_FT_2025-01-24-10-08-42/best_checkpoint.pth'
    #             # else: 
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/EffNetB4_thesis_no_occ_FT/training/EfficientNet_B4_FT_2024-10-18-11-44-29/best_checkpoint.pth'
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'gotcha_occ': 
    #         elif 'gotcha_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = './results/results_FT/EffNetB4_gotcha_occ_FT_data_aug_new/training/EfficientNet_B4_FT_2025-01-24-18-42-25/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     print("no crop data aug model!")
    #             #     exit()
    #             #     # pretrained_model_path = './results/results_FT/EffNetB4_gotcha_occ_FT_data_aug_crop/training/EfficientNet_B4_FT_2025-01-24-13-35-50/best_checkpoint.pth'
    #             # else: 
    #             # pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/EffNetB4_gotcha_occ_FT/training/EfficientNet_B4_FT_2025-03-05-09-49-48/best_checkpoint.pth' # test with adam opt
    #                 #'/home/rz/rz-test/bceWLL_test/results/results_FT/EffNetB4_gotcha_occ_FT_Adam_opt/training/EfficientNet_B4_FT_2025-02-17-16-27-07/best_checkpoint.pth'
                
    #             # ------------------------------------------ #
    #             #[original best model]  pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/EffNetB4_gotcha_occ_FT/training/EfficientNet_B4_FT_2024-10-21-11-29-56/best_checkpoint.pth'
    #             # ------------------------------------------ #
    #             # test with Adam optimizer
    #             pretrained_model_path = './results/results_FT/EffNetB4_gotcha_occ_ADAM_FT/training/EfficientNet_B4_FT_2025-03-06-09-12-54/best_checkpoint.pth'
                
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])

    #         # elif args.dataset == 'gotcha_no_occ':
    #         elif 'gotcha_no_occ' in args.model:
    #             # pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/EffNetB4_gotcha_no_occ_FT/training/EfficientNet_B4_FT_2024-10-21-12-21-19/best_checkpoint.pth'
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = './results/results_FT/EffNetB4_gotcha_no_occ_FT_data_aug_new/training/EfficientNet_B4_FT_2025-01-24-20-13-51/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = 'UPDATE_PATH!!'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else:
    #                 # retrained model
    #             # pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/EffNetB4_gotcha_no_occ_FT/training/EfficientNet_B4_FT_2025-03-05-11-31-37/best_checkpoint.pth' # test with Adam opt
                    
    #                 # 
    #                 # '/home/rz/rz-test/bceWLL_test/results/results_FT/EffNetB4_gotcha_no_occ_FT_Adam_opt/training/EfficientNet_B4_FT_2025-02-17-17-29-09/best_checkpoint.pth''


    #             # ------------------------------------------ #
    #             #  [AdamW best model] pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/EffNetB4_gotcha_no_occ_FT/training/EfficientNet_B4_FT_2025-01-21-12-27-51/best_checkpoint.pth'
    #             # ------------------------------------------ #
    #             # test with adam optimizer
    #             pretrained_model_path = './results/results_FT/EffNetB4_gotcha_no_occ_ADAM_FT/training/EfficientNet_B4_FT_2025-03-06-10-08-39/best_checkpoint.pth'
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])

    #     else:
    #         print("using pretrained ImageNet model")
    #         pretrained_model_path = 'EfficientNet_B4 Pre-trained ImageNet Model'
    #         # Replace the classifier layer
    #         # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
    #         model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
    #         # check_model_gradient(model)
    #         # 
    #     # pretrained_model_path = args.save_model_path 
        
    #     model.to(device)
    #     print("Model loaded!")

    #     # for key in model.keys():
    #     #     print(key)

    #     # print(model)
    #     # exit()
        
    # # elif args.model == 'xception':
    # elif 'xception' in args.model.lower():
    #     print("Loading pretrained XceptionNet model...")
    #     # load the xceptionet model
    #     # pip install timm
    #     # import timm
    #     model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
    #     model_name = 'XceptionNet' # add the model name to the model object
    #     # load the pre-trained model for testing (load model weights)
    #     # pretrained_model_path = args.save_model_path 
    #     if args.tl:
    #         # ---------------------------- #
    #         # --- TRANSFER LEARNING!!! --- #
    #         # ---------------------------- #
    #         print("loading pretrained model")
    #         # if args.dataset == 'thesis_occ':
    #         if 'thesis_occ' in args.model:
    #             # load the pre-trained model for testing (load model weights)
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/TL/results_TL/Xception_thesis_occ_TL/training/model/best_model.pth'
    #             # '/home/rz/rz-test/bceWLL_test/model/XceptionNet_baseline_occ/best_model.pth'  #args.save_model_path 
    #             print("model saved in:", pretrained_model_path)
    #             model.load_state_dict(torch.load(pretrained_model_path))
    #             # best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             # print(best_ckpt)
    #             # 
    #             # model.load_state_dict(best_ckpt['model']) 
    #         # elif args.dataset == 'thesis_no_occ':
    #         elif 'thesis_no_occ' in args.model:
    #             pretrained_model_path = pretrained_model_path = '/media/data/model_exp_results/face_occ_net/TL/results_TL/Xception_thesis_no_occ_TL/training/model/best_model.pth'
    #             #'/home/rz/rz-test/bceWLL_test/model/XceptionNet_baseline_no_occ/best_model_2.pth'
    #             print("model saved in:", pretrained_model_path)
    #             model.load_state_dict(torch.load(pretrained_model_path)) 
    #         # elif args.dataset == 'gotcha_occ':
    #         elif 'gotcha_occ' in args.model:
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/TL/results_TL/Xception_gotcha_occ_TL/training/XceptionNet_2024-10-08-12-18-20/best_checkpoint.pth'
    #             print("model saved in:", pretrained_model_path)
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         # elif args.dataset == 'gotcha_no_occ':
    #         elif 'gotcha_no_occ' in args.model:
    #             pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_TL/Xception_gotcha_no_occ_TL/training/XceptionNet_2025-02-26-17-22-40/best_checkpoint.pth'
    #             #'/media/data/model_exp_results/face_occ_net/TL/results_TL/Xception_gotcha_no_occ_TL/training/XceptionNet_2024-10-08-13-23-24/best_checkpoint.pth'
    #             print("model saved in:", pretrained_model_path)
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         else:
    #             print("no pretrained model found")
    #             exit()

    #     elif args.ft:
    #         # ---------------------- #
    #         # --- FINE TUNING!!! --- #
    #         # ---------------------- #
    #         # if args.dataset == 'thesis_occ':
    #         if 'thesis_occ' in args.model:
                
    #             # # retrained model
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_thesis_occ_FT_data_aug_new/training/XceptionNet_2025-01-25-16-23-35/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = 'UPDATE_PATH!!'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else:
    #                 # load the pre-trained model for testing (load model weights)
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/Xception_thesis_occ_FT/training/XceptionNet_2024-10-18-12-57-51/best_checkpoint.pth'
                
    #             print("model saved in:", pretrained_model_path)
    #             # model.load_state_dict(torch.load(pretrained_model_path))
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model']) 

    #         # elif args.dataset == 'thesis_no_occ':
    #         elif 'thesis_no_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_thesis_no_occ_FT_data_aug_new/training/XceptionNet_2025-01-25-16-53-43/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = 'UPDATE_PATH!!'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else:
    #                 # load the pre-trained model for testing (load model weights)
    #             pretrained_model_path = pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/Xception_thesis_no_occ_FT/training/XceptionNet_2024-10-18-13-17-14/best_checkpoint.pth'
    #             # retrained model
                
    #             print("model saved in:", pretrained_model_path)
    #             # model.load_state_dict(torch.load(pretrained_model_path))
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model']) 

    #         # elif args.dataset == 'gotcha_occ':
    #         elif 'gotcha_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_gotcha_occ_FT_data_aug_new/training/XceptionNet_2025-01-25-17-08-50/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = 'UPDATE_PATH!!'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else:
    #                 # load the pre-trained model for testing (load model weights)
    #             # pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/Xception_gotcha_no_occ_FT/training/XceptionNet_2025-01-21-13-49-19/best_checkpoint.pth'
    #                 # '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_gotcha_occ_FT_Adam_opt/training/XceptionNet_2025-02-17-18-26-20/best_checkpoint.pth'
    #             # retrained model
                
    #             # ------------------------------------------ #
    #             # best AdamW model #
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/Xception_gotcha_occ_FT/training/XceptionNet_FT_2024-10-21-13-21-56/best_checkpoint.pth'
    #             #'/media/data/model_exp_results/face_occ_net/results_FT/Xception_gotcha_no_occ_FT/training/XceptionNet_2025-01-21-13-49-19/best_checkpoint.pth'
    #             # ------------------------------------------ #
    #             # test with Adam optimizer
    #             # pretrained_model_path = './results/results_FT/Xception_gotcha_occ_ADAM_FT/training/XceptionNet_2025-03-06-10-45-41/best_checkpoint.pth'
    #             # ------------------------------------------ #
    #             print("model saved in:", pretrained_model_path)
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])

    #         # elif args.dataset == 'gotcha_no_occ':
    #         elif 'gotcha_no_occ' in args.model:
    #             # if args.data_aug == 'new':
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_gotcha_occ_FT_data_aug_new/training/XceptionNet_2025-01-25-17-08-50/best_checkpoint.pth'
    #             # elif args.data_aug == 'crop':
    #             #     # pretrained_model_path = 'UPDATE_PATH!!'
    #             #     print("no CenterCrop data aug model!")
    #             #     exit()
    #             # else:
    #             #     # load the pre-trained model for testing (load model weights)
    #             #     pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_gotcha_no_occ_FT_Adam_opt/training/XceptionNet_2025-02-17-20-28-26/best_checkpoint.pth'
    #                 #'/media/data/model_exp_results/face_occ_net/results_FT/Xception_gotcha_no_occ_FT/training/XceptionNet_2025-01-21-13-49-19/best_checkpoint.pth'
    #             # retrained model
    #             # pretrained_model_path = './results/results_FT/Xception_gotcha_no_occ_FT/training/XceptionNet_2025-01-21-13-49-19/best_checkpoint.pth'
    #             # pretrained_model_path = '/home/rz/rz-test/bceWLL_test/results/results_FT/Xception_gotcha_no_occ_FT_data_aug_new/training/XceptionNet_2025-01-25-18-32-11/best_checkpoint.pth'
                
    #             # ------------------------------------------ #
    #             # best AdamW model #
    #             pretrained_model_path = '/media/data/model_exp_results/face_occ_net/results_FT/Xception_gotcha_no_occ_FT/training/XceptionNet_2025-01-21-13-49-19/best_checkpoint.pth'
    #             # ------------------------------------------ #
    #             # test with Adam optimizer
    #             # pretrained_model_path = './results/results_FT/Xception_gotcha_no_occ_ADAM_FT/training/XceptionNet_2025-03-06-11-51-53/best_checkpoint.pth'
    #             # ------------------------------------------ #
    #             print("model saved in:", pretrained_model_path)
    #             best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #             model.load_state_dict(best_ckpt['model'])
    #         else:
    #             print("no pretrained model found")
    #             exit()

    #     else:
    #         print("using pretrained ImageNet model")
    #         pretrained_model_path = 'Xception Pre-trained ImageNet Model'
    #         # check_model_gradient(model)
    #         # 
            
        
    #     model.to(device)
    #     print("Model loaded!")

    #     # print(model)
    #     # check_model_gradient(model)
    #     # 

    # else:
    #     print("Model not supported")
    #     exit()

    # print("model path: ", pretrained_model_path)

    model, pretrained_model_path = load_model_from_path(args, device)
    model.to(device)
    print("Model loaded!")
    print("model_path: ", pretrained_model_path)
    print(model)

    # breakpoint()
    

    # load the dataset -> for now only the face occlusion dataset (thesis), later add the GOTCHA dataset
    # NOTE: also need to fix the customDataset.py to load the GOTCHA dataset as well!!
    # dataset
    gotcha = False
    if args.dataset == 'fows_occ':
        # train_dir = '/media/data/rz_dataset/users_face_occlusion/training/'
        test_dir = '/media/data/rz_dataset/users_face_occlusion/testing/'
        
    elif args.dataset == 'fows_no_occ':
        # train_dir = '/media/data/rz_dataset/user_faces_no_occlusion/training/'
        test_dir = '/media/data/rz_dataset/user_faces_no_occlusion/testing/'
        
    elif args.dataset == 'milan_occ':
        test_dir = '/media/data/rz_dataset/milan_faces/occlusion/testing/'
        args.data_aug = 'milan'

    elif args.dataset == 'milan_no_occ':
        test_dir = '/media/data/rz_dataset/milan_faces/no_occlusion/testing/'
        args.data_aug = 'milan'
        # testing -> user_300229  user_792539
        # training -> all the other users

    # GOTCHA dataset -> might need to modify the dataloader since gotcha has a slightly different structure than the other datasets
    # need to check also the gotcha dataset -> divide images into occlusion and no occlusion
    elif args.dataset == 'gotcha_occ':
        test_dir = '/media/data/rz_dataset/gotcha/balanced_gotcha/occlusion/testing'
        gotcha = True

    elif args.dataset == 'gotcha_no_occ':
        test_dir = '/media/data/rz_dataset/gotcha/balanced_gotcha/no_occlusion/testing'
        gotcha = True

    # @TODO: add the GOTCHA dataset here and in the customDataset.py for the dataloader

    else:
        print("Dataset not supported")
        exit()

    print("dataset:", args.dataset)
    print("test_dir:", test_dir)
    
    # --------------------------------- #
    # Define the dataset and dataloaders
    # --------------------------------- #
    print("batch_size: ", args.batch_size)
    print("data_aug: ", args.data_aug)
    

    if args.data_aug:
        print("Using default data augmentation (thesis)")

        # Transformations for testing data
        test_transform = transforms.Compose([
            transforms.Resize((256,256)), 
            # BICUBIC is used for EfficientNetB4 -> check the documentation
            # BICUBIC vs BILINEAR -> https://discuss.pytorch.org/t/what-is-the-difference-between-bilinear-and-bicubic-interpolation/20920 
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # test dataloader 
        # test_dataset = FaceImagesDataset(test_dir, test_transform)
        test_dataset = FaceImagesDataset(dataset_dir=test_dir, transform=test_transform, use_albu=False, training = False) #, challenge=args.trn_challenge, algo=args.trn_algo)
        print("test_dataset: ", len(test_dataset))
        # get the test_dataset labels
        dataset_labes = test_dataset.label_map
        print(dataset_labes)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset) # load the dataset in the order it is in the folder
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size, #64, 
                                    sampler=sampler_test,
                                    num_workers = 3) # shuffle=False)

    else:
        print("Data augmentation not supported")
        exit()

    print("data augmentations: ", args.data_aug)
    print(test_transform)
    # 

    # -------------------------------------------------------- #
    # Testing the model
    # -------------------------------------------------------- #

    # print(f'Testing the pretrained model {model_name} on the {args.dataset} dataset...')
    print(f'Testing the pretrained model {args.model} on the {args.dataset} dataset...')
    if args.wandb:
            run.tags = run.tags + ('Testing-model',) # add testing tag

    # creating a folder where to save the testing results 
    if args.robust and args.tags:
        exp_results_path = f'./results/results_robusteness_test/{args.tags}/testing'
    elif args.tl:
            exp_results_path = f'./results/results_TL/{args.tags}/testing' # i.e. ./results/EfficientNetB4_FF_no_occ_focal_loss/testing
    elif args.ft:
            exp_results_path = f'./results/results_FT/{args.tags}/testing'
    elif args.gen:
        exp_results_path = f'./results/results_GEN/{args.tags}/testing'
    else: 
        exp_results_path = f'./results/results/{args.tags}/testing'

    # Evaluate the model on the test data
    if not gotcha:
        # test_accuracy, balanced_test_accuracy, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, labels_collection, prediction_collection, img_path_collection = test_one_epoch(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh) 
        # test_accuracy, balanced_test_acc, test_accuracy_ original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, list(labels_list), list(probs_list), img_path_collection
        

        # test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_simswap, \
        # test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, \
        # prob_original, prob_simswap, prob_ghost, prob_facedancer, auc_best_threshold, eer_threshold 
        # labels_collection, img_path_collection, prediction_collection 
        
        test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold = test_one_epoch(
            model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh
        )

        # print(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
        print(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
        print("Dataset labels: ", dataset_labes)
        print(f"Data augmentation: {args.data_aug}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
        print(f"Test Accuracy Original: {test_accuracy_original:.4f}")
        print(f"Test Accuracy SimSwap: {test_accuracy_simswap:.4f}")
        print(f"Test Accuracy Ghost: {test_accuracy_ghost:.4f}")
        print(f"Test Accuracy FaceDancer: {test_accuracy_facedancer:.4f}")
        
        # print(f"AUC Score: {auc:.4f}") 
        print(f"Average Precision: {ap_score:.4f}")
        print(f"TPR (sensitivity): {TPR:.4f}")
        print(f"TNR (specificity): {TNR:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"AUC thresh: {auc_best_threshold:.4f}")
        print(f"EER: {eer:.4f}")
        print(f"EER thresh: {eer_threshold:.4f}")

        # log results to wandb
        if args.wandb:
            run.tags = run.tags + ('Testing',) # add testing tag
            
            run.log({'Test Accuracy': test_accuracy})
            run.log({'Balanced Test Accuracy': balanced_test_acc})
            run.log({'Test Accuracy Original': test_accuracy_original}) 
            run.log({'Test Accuracy SimSwap': test_accuracy_simswap}) 
            run.log({'Test Accuracy Ghost': test_accuracy_ghost})
            run.log({'Test Accuracy FaceDancer': test_accuracy_facedancer})
            
            run.log({'AP': ap_score})
            run.log({'TPR': TPR})
            run.log({'TNR': TNR})
            run.log({'AUC': auc_score})
            run.log({'EER': eer})
            

        # log the results to a log file
        if args.save_log:
            print(f"logging results - {args.tags}")
            # set up logger
            log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
            print("log_path: ", log_path)
            log = create_logger(log_path)
            # log.info("")
            # log.info(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
            log.info(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
            log.info(f"Dataset test dir: {test_dir}")
            log.info(f"Model weights: {pretrained_model_path}")
            log.info(f"FT: {args.ft}, TL: {args.tl}, GEN: {args.gen}")
            log.info(f"Dataset labels: {dataset_labes}")
            log.info(f"Data augmentation: {args.data_aug}")
            log.info(f"Test Accuracy: {test_accuracy:.4f}")
            log.info(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
            log.info(f"Test Accuracy Original: {test_accuracy_original:.4f}")
            log.info(f"Test Accuracy SimSwap: {test_accuracy_simswap:.4f}")
            log.info(f"Test Accuracy Ghost: {test_accuracy_ghost:.4f}")
            log.info(f"Test Accuracy FaceDancer: {test_accuracy_facedancer:.4f}")
            
            # log.info(f"AUC: {auc:.4f}")
            log.info(f"Average Precision: {ap_score:.4f}")
            log.info(f"TPR (recall): {TPR:.4f}")
            log.info(f"TNR (specificity): {TNR:.4f}")
            log.info(f"AUC: {auc_score:.4f}")
            log.info(f"AUC thresh: {auc_best_threshold:.4f}")
            log.info(f"EER: {eer:.4f}")
            log.info(f"EER thresh: {eer_threshold:.4f}")
            # log.info(f"TPR: {list(tpr_roc)}")
            # log.info(f"FPR: {list(fpr_roc)}")
            # log.info(f"Specificity (TNR): {list(specificity)}")
            # log.info(f"Precision: {precision_scr:.4f}")
            # log.info(f"Recall: {list(recall)}")
            log.info("Note the labels/preds/imgs_paths are in the json file")
            # log.info(f"Labels collection: {labels_collection}")
            # log.info(f"Prediction collection: {prediction_collection}")
            # log.info(f"Imgs paths collection: {img_path_collection}")
            # log.info(f"prob_original: {prob_original}")
            # log.info(f"prob_facedancer: {prob_facedancer}")
            # log.info(f"prob_ghost: {prob_ghost}")
            # log.info(f"prob_simswap: {prob_simswap}")

    else: 
        print("Testing GOTCHA dataset!")

        # test_accuracy, balanced_test_accuracy, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, labels_collection, prediction_collection, img_path_collection = gotcha_test(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh) 
        
        # test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, \
        # TPR, TNR, auc_score, ap_score, eer, labels_list, img_path_collection, prob_original, \
        # prob_dfl, prob_fsgan, auc_best_threshold, eer_threshold 
        #  img_path_collection, labels_collection, prediction_collection,
        test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold = gotcha_test(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh)
        
        # print(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
        print(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
        print("Dataset labels: ", dataset_labes)
        print(f"Dataset test dir: {test_dir}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Balanced Test Accuracy: {balanced_test_acc:.4f}")
        print(f"Test Accuracy Original: {test_accuracy_original:.4f}")
        print(f"Test Accuracy DFL: {test_accuracy_dfl:.4f}")
        print(f"Test Accuracy FSGAN: {test_accuracy_fsgan:.4f}")
        # print(f"Test Accuracy FaceDancer: {test_accuracy_facedancer:.4f}")
        
        # print(f"AUC Score: {auc:.4f}") 
        print(f"Average Precision: {ap_score:.4f}")
        print(f"TPR (sensitivity): {TPR:.4f}")
        print(f"TNR (specificity): {TNR:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"AUC thresh: {auc_best_threshold:.4f}")
        print(f"EER: {eer:.4f}")
        print(f"EER thresh: {eer_threshold:.4f}")

        # log results to wandb
        if args.wandb:
            run.tags = run.tags + ('Testing',) # add testing tag
            
            run.log({'Test Accuracy': test_accuracy})
            run.log({'Balanced Test Accuracy': balanced_test_acc})
            run.log({'Test Accuracy Original': test_accuracy_original}) 
            run.log({'Test Accuracy DFL': test_accuracy_dfl}) 
            run.log({'Test Accuracy FSGAN': test_accuracy_fsgan})
            
            run.log({'AP': ap_score})
            run.log({'TPR': TPR})
            run.log({'TNR': TNR})
            run.log({'AUC': auc_score})
            run.log({'EER': eer})
            

        # log the results to a log file
        if args.save_log:
            print(f"logging results - {args.tags}")
            # set up logger
            log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
            log = create_logger(log_path)
            # log.info("")
            # log.info(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
            log.info(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
            log.info(f"Model weights: {pretrained_model_path}")
            log.info(f"Dataset labels: {dataset_labes}")
            log.info(f"Dataset test dir: {test_dir}")
            log.info(f"Data augmentation: {args.data_aug}")
            log.info(f"Test Accuracy: {test_accuracy:.4f}")
            log.info(f"Balanced Test Accuracy: {balanced_test_acc:.2f}")
            log.info(f"Test Accuracy Original: {test_accuracy_original:.4f}")
            log.info(f"Test Accuracy DFL: {test_accuracy_dfl:.4f}")
            log.info(f"Test Accuracy FSGAN: {test_accuracy_fsgan:.4f}")
            
            # log.info(f"AUC: {auc:.4f}")
            log.info(f"Average Precision: {ap_score:.4f}")
            log.info(f"TPR (recall): {TPR:.4f}")
            log.info(f"TNR (specificity): {TNR:.4f}")
            log.info(f"AUC: {auc_score:.4f}")
            log.info(f"EER: {eer:.4f}")
            # log.info(f"TPR: {list(tpr_roc)}")
            # log.info(f"FPR: {list(fpr_roc)}")
            # log.info(f"Specificity (TNR): {list(specificity)}")
            # log.info(f"Precision: {precision_scr:.4f}")
            # log.info(f"Recall: {list(recall)}")
            log.info("Note the labels/preds/imgs_paths are in the json file")
            # log.info(f"Labels collection: {labels_collection}")
            # log.info(f"Prediction collection: {prediction_collection}")
            # log.info(f"Imgs paths collection: {img_path_collection}")
            # log.info(f"prob_original: {prob_original}")
            # log.info(f"prob_dfl: {prob_dfl}")
            # log.info(f"prob_fsgan: {prob_fsgan}")
            log.info("")

    # close wandb session
    if args.wandb:
        run.finish() # end the wandb session

if __name__ == '__main__':
    main()