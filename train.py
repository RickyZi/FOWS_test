# ---------------------------------------------- #
# code for training the models - to be completed #
# ---------------------------------------------- #

"""
ideally it should be a "summary" of all training codes for the models
    - mnetv2
    - effnetb4
    - xception
    - icpr2020 (Milan_EffNetB4)
    - Neurips2023 (DFB_xceptionNet)

# ------------------------------------------------------------------- #
NOTE:
 
"""
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
# from utils.gotcha_trn_val_tst import *
# import cv2
import wandb # for logging results to wandb
import argparse # for command line arguments
# pip install efficientnet_pytorch # need to install this package to use EfficientNet
# from efficientnet_pytorch import EfficientNet

from utils.logger import * # import the logger functions
import timm # for using the XceptionNet model (pretrained)
# pip install timm # install the timm package to use the XceptionNet model
# import utils.papers_data_augmentations
from utils.papers_data_augmentations import *

import utils.fornet as fornet
from utils.fornet import *

import yaml
# from dfb.dfb_detectors import DETECTOR

import random
import datetime

from utils.focalLoss import FocalLoss 
# ------------------------------------------------------------------------------------------- #
# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
   
    parser.add_argument('--model', type=str, default='mnetv2', help='Model to use for training and testing')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') # default for adam and adamw
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use for training')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for the optimizer') # default for adamw
    parser.add_argument('--patience', type=int, default=3, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--loss', type=str, default='focal', help='Loss function to use for training')
    parser.add_argument('--thresh', type = float, default = 0.5, help = 'Threshold for binary classification.')
    parser.add_argument('--save-log', action='store_false', help='Save the model output logs')
    parser.add_argument('--save-model', action = 'store_false', help='Save the model in a folder with the same name as the training tags.')
    parser.add_argument('--data-aug', type=str, default='default', help='Data augmentation to use for training and testing') 
    parser.add_argument('--tl', action='store_true', help='Use transfer learning for the model') # use transfer learning for the model
    parser.add_argument('--ft', action='store_true', help='Fine-Tuning the model') # use transfer learning for the model
    parser.add_argument('--dataset', type=str, default='thesis_occ', help='Path to the training and/or testing dataset')
    parser.add_argument('--resume', type = str, help = 'resume training from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging') # if not specified in the command as --wandb the value is set to False
    parser.add_argument('--tags', type=str, default='face-occlusion', help='Info about the model, training setting, dataset, test setting, etc.')

    return parser


def check_model_name(model_path):
    i = 1
    while os.path.exists(model_path): # check if the new graph name already exists -> if it does, increment i 
        # while stops when the new graph name does not exist 
        # new_graph_name = f'{graph_name}_{i}.png'
        i += 1
        model_name = model_path.split('/')[-1].split('.')[0] # get the model name from the model path (remove .pth)
        model_path = model_path.replace(model_name, f'{model_name}_{i}') # replace the model name with the new model name
    return model_path

def milan_save_model(net: nn.Module, optimizer: optim.Optimizer,
               train_loss: float, val_loss: float,
               iteration: int, batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


def init_seed():
    # --------------------------------------------- #
    # fnct from DFB code
    # def init_seed(config):
    #     if config['manualSeed'] is None:
    #         config['manualSeed'] = random.randint(1, 10000)
    #     random.seed(config['manualSeed'])
    #     torch.manual_seed(config['manualSeed'])
    #     if config['cuda']:
    #         torch.cuda.manual_seed_all(config['manualSeed'])
    # Set the random seed for reproducibility
    random.seed(42)
    # np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
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

def main(): 
    # ----------------------------------------- #
    # main function to train and test the model #
    # ----------------------------------------- #


    # Parse the arguments
    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    pretrained_model_path = ''
    # --------------------------------- #
    # wandb setup
    # --------------------------------- #
    # if args.wandb:
    #     # start the wandb session
    #     run = wandb.init(
    #             project='gotcha', # created new project to track the results of the model
    #             entity ="dl-ais",
    #             # config={
    #             #     "learning_rate": 0.001,
    #             #     "architecture": "mobilenetv2",
    #             #     "dataset": "face-occlusion-dirty",
    #             #     "epochs": 10,
    #             #     "loss": "BCEwithLogitsLoss",
    #             #     },
    #             # tags =["new_focal_loss", "no_resize_rotation_color_jitter"]
    #             tags=[args.tags], # for finding the wandb session online
    #             # id = args.tags # set the id of the wandb session to the tags
    #     )
    #     # run_id = run.id # use id to get output.log file of the run
    #     # print(run.id) # print the wandb session id 
    #     # ./wandb/run-20240610_124102-zppjwmnl
    #     # run-20240610_124102-zppjwmnl -> run id
    # else:
    #     print("Not using wandb for logging")


    # ---------------------------------------------------------- #
    # Load the pre-trained model (MobileNetV2 or EfficientNetB4) #
    # ---------------------------------------------------------- #   

    init_seed() # init random seed for reproducibility

    # if args.scratch and args.model == 'mobilenetv2':
    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # milan_model = False
    print("thresh: ", args.thresh)
    # ----------------------- #
    # --- BASELINE MODELS --- #
    # ----------------------- #

    if args.model == 'mnetv2': 
        print("Loading MobileNetV2 model")

        if args.tl and args.resume:
            print("Resume training from checkpoint for MobileNetV2")
            print("checkpoint path:", args.resume)
            pretrained_model_path = args.resume
            model_name = "MobileNetV2_TL"
            checkpoint = torch.load(args.resume, map_location = "cpu")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            model.load_state_dict(checkpoint['model'])
            # model.to(device)
            print('Model loaded!')

        elif args.tl:
            print("Loading MobileNetV2 pre-trained model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

            # check the model gradient
            check_model_gradient(model)
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
            check_model_gradient(model)
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
            check_model_gradient(model)
            # ()
            model_name = 'MobileNetV2' # add the model name to the model object

        model.to(device)
        print("Model loaded!")

        # print(model)
        # ()
        # exit()

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
            
            if args.resume:
                print("Resume training from checkpoint for EfficientNetB4")
                checkpoint = torch.load(args.resume, map_location = "cpu")
                pretrained_model_path = args.resume
                model.load_state_dict(checkpoint['model'])
                # if 'optimizer' in checkpoint and 'epoch' in checkpoint and 'criterion' in checkpoint:
                #     optimizer.load_state_dict(checkpoint['optimizer'])
                #     # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                #     start_epoch = checkpoint['epoch'] +1 
                #     criterion = checkpoint['criterion']

                print("model loaded!")
                print("model info:")
                print("epoch: ", checkpoint['epoch'])
                # print("train_loss: ", checkpoint['train_loss'])
                print("val_loss: ", checkpoint['val_loss'])
                # print("train_accuracy: ", checkpoint['train_accuracy'])
                print("val_accuracy: ", checkpoint['val_accuracy'])
                print("optimizer: ", checkpoint['optimizer'])
                print("criterion: ", checkpoint['criterion'])
                print("last_epoch: ", checkpoint['epoch'])

            model_name = 'EfficientNet_B4' # add the model name to the model object
            model.to(device)
            # check the model gradient
            check_model_gradient(model)
            # ()
        # ------------------------------------------------------------------------ #
        elif args.ft: 
            print("Fine-Tuning the EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real or swap face
            model_name = 'EfficientNet_B4_FT'

            if args.resume:
                print("Resume training from checkpoint for EfficientNetB4")
                print("checkpoint path:", args.resume)
                pretrained_model_path = args.resume

                # check if checkpoint file is ok
                if os.path.isfile(args.resume):
                    try:
                        checkpoint = torch.load(args.resume, map_location="cpu")
                        print("Checkpoint loaded successfully.")
                    except RuntimeError as e:
                        print(f"Error loading checkpoint: {e}")
                else:
                    print(f"Checkpoint file {args.resume} does not exist.")

                checkpoint = torch.load(args.resume, map_location = "cpu")
                checkpoint.keys() # check the keys in the checkpoint
                # ()
                model.load_state_dict(checkpoint['model'])
                print("model loaded!")
                print("model info:")
                print("epoch: ", checkpoint['epoch'])
                # print("train_loss: ", checkpoint['train_loss'])
                print("val_loss: ", checkpoint['val_loss'])
                # print("train_accuracy: ", checkpoint['train_accuracy'])
                print("val_accuracy: ", checkpoint['val_accuracy'])
                print("optimizer: ", checkpoint['optimizer'])
                print("criterion: ", checkpoint['criterion'])
                print("last_epoch: ", checkpoint['epoch'])
            
            model.to(device)
            # check the model gradient
            check_model_gradient(model)
            # ()

            
        else: 
            # print("problem in loading the model")
            # exit()
            print("Loading EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            model.to(device)
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
           
            # check the model gradient
            check_model_gradient(model)

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

            print('Model loaded!')
            model.to(device)

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
            check_model_gradient(model)
            # ()

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
            check_model_gradient(model)
        # else:
        #     # retraining the model with one output class 
        #     model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real or swap face
        #     model_name = 'XceptionNet' # add the model name to the model object

        # ()

        model.to(device)
        print("Model loaded!")

        print(model)
        # exit()
        # check the model gradient
        check_model_gradient(model)
        # ()


    # ------------------------ #
    # ---- ICPR2020 MODEL ---- #
    # ------------------------ #
    elif args.model == 'icpr2020':
        print("Loading ICPR2020 model")
        args.optimizer = "adam"
        if args.tl and args.resume:
            print("Resume training from checkpoint for EfficientNetB4_DFDC")
            print("checkpoint path:", args.resume)
            pretrained_model_path = args.resume
            model_name = "EfficientNetB4_DFDC_TL"
            checkpoint = torch.load(args.resume, map_location = "cpu")
            net_name = "EfficientNetB4"
            net_class = getattr(fornet, net_name)
            # net: FeatureExtractor = net_class().eval().to(device)
            model: FeatureExtractor = net_class().to(device)
            # incomp_keys = model.load_state_dict(model_state['net'], strict=True)
            incomp_keys = model.load_state_dict(checkpoint['model'], strict=True)
            print(incomp_keys)
            # print(model)
            print('Model loaded!')

        elif args.tl:
            print("Loading EfficientNetB4_DFDC pre-trained model")
            pretrained_model_path = './_pretrained_models_papers_/icpr2020dfdc_weights/EfficientNetB4_DFDC/bestval.pth'
            model_state = torch.load(pretrained_model_path, map_location = "cpu")
            # milan_model = True
            # net_name = "EfficientNetB4"
            net_class = getattr(fornet, "EfficientNetB4")
            # net: FeatureExtractor = net_class().eval().to(device)
            model: FeatureExtractor = net_class().to(device)
            incomp_keys = model.load_state_dict(model_state['net'], strict = True)
            model_name = "EfficientNetB4_DFDC"
            # incomp_keys = net.load_state_dict(state['net'], strict=True)
            print(incomp_keys)
            print('Model loaded!')
            print("Transfer learning - Freezing all layers except the classifier")
            # Freeze all layers except the classifier
            for param in model.parameters():
                param.requires_grad = False

            # Ensure the classifier layer is trainable
            for param in model.classifier.parameters():
                param.requires_grad = True

        else: 
            # print("problem in loading the model")
            # exit()
            print("Loading EfficientNetB4_DFDC pre-trained model")
            pretrained_model_path = './_pretrained_models_papers_/icpr2020dfdc_weights/EfficientNetB4_DFDC/bestval.pth'
            model_state = torch.load(pretrained_model_path, map_location = "cpu")
            # milan_model = True
            # net_name = "EfficientNetB4"
            net_class = getattr(fornet, "EfficientNetB4")
            # net: FeatureExtractor = net_class().eval().to(device)
            model: FeatureExtractor = net_class().to(device)
            incomp_keys = model.load_state_dict(model_state['net'], strict = True)
            model_name = "EfficientNetB4_DFDC"
            # incomp_keys = net.load_state_dict(state['net'], strict=True)
            print(incomp_keys)
            print('Model loaded!')

    else:
        print("Model not supported")
        exit()


    # # Move the model to GPU if available
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # print("device", device) # output: cpu -> gpu drivers not active/updated??
    # model = model.to(device)


    # load the dataset -> for now only the face occlusion dataset (thesis), later add the GOTCHA dataset
    # NOTE: also need to fix the customDataset.py to load the GOTCHA dataset as well!!
    # dataset
    gotcha = False
    if args.dataset == 'thesis_occ':
        train_dir = '/media/data/rz_dataset/users_face_occlusion/training/'
        # test_dir = '/media/data/rz_dataset/users_face_occlusion/testing/'
        # args.data_aug = ''
        

    elif args.dataset == 'thesis_no_occ':
        train_dir = '/media/data/rz_dataset/user_faces_no_occlusion/training/'
        # test_dir = '/media/data/rz_dataset/user_faces_no_occlusion/testing/'

    elif args.dataset == 'milan_occ':
        train_dir = '/media/data/rz_dataset/milan_faces/occlusion/training'
        args.data_aug = 'milan'

    elif args.dataset == 'milan_no_occ':
        train_dir = '/media/data/rz_dataset/milan_faces/no_occlusion/training'
        args.data_aug = 'milan'

    elif args.dataset == 'gotcha_occ':
        train_dir = '/media/data/rz_dataset/gotcha/balanced_gotcha/occlusion/training'
        gotcha = True
        # num-epochs 15 --batch-size 64 --optimizer "adamw" --loss "bce"
        args.num_epochs = 15
        args.loss = "bce"
        # if args.model not in ["effnetb4_dfdc", "effnetb4_ff"]:
        #     args.optimizer = 'adamw'

    elif args.dataset == 'gotcha_no_occ':
        train_dir = '/media/data/rz_dataset/gotcha/balanced_gotcha/no_occlusion/training'
        gotcha = True
        args.num_epochs = 15
        args.loss = "bce"
        # if args.model not in ["effnetb4_dfdc", "effnetb4_ff"]:
        #     args.optimizer = 'adamw'

    else:
        print("Dataset not supported")
        exit()

    print("dataset:", args.dataset)
    print("train_dir", train_dir)
    # print("train_dir:", train_dir)
    # print("test_dir:", test_dir)



    num_epochs = args.num_epochs
    print("num_epochs: ", args.num_epochs)
    # lr = args.lr # 0.001 = 1e-3 (default value)
    # wheight_decay = args.weight_decay # 1e-5 (default value)
    patience = args.patience # 3 (number of epochs to wait for improvement before stopping)
    best_val_loss = float('inf')  # Initialize best validation loss
    early_stopping_counter = 0  # Counter to keep track of non-improving epochs

    # --------------------------------- #
    # Define the optimizer #
    # --------------------------------- #

    if args.optimizer == 'adam':
        if model_name in ["effnetb4_ff", "effnetb4_dfdc"]: #, "EfficientNetB4_DFB", "XceptionNet_DFB", "UCF_DFB"]:
            optimizer = optim.Adam(model.get_trainable_parameters(), lr = args.lr)
        else: optimizer = optim.Adam(model.parameters(), lr= args.lr) #, weight_decay= args.weight_decay) # add weight decay

    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr= args.lr, weight_decay= args.weight_decay) 
        # default values:
        # lr = 1e-3 ()
        # weight_decay = 1e-2 (try with differnet values, 1e-5, 1e-4, 1e-3)

    else: 
        print("Optimizer not supported")
        exit()
        
    print("optimizer: ", args.optimizer)
    # breakpoint()
    # --------------------------------- #
    # Define the loss function #
    # Focal loss
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2) 
        # def values from pytorch documentation (https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html)
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss() # used by milan models
    elif args.loss == 'bce_loss':
        criterion = nn.BCELoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Loss function not supported")
        exit()

    print("Loss Function: ", criterion)
    # ---------------------------------- #

    if args.save_model and args.tl:
        # model_path = './model/' + args.tags +'_train' + '/' + 'best_model.pth'
        save_model_folder = f'./results/results_TL/{args.tags}/training/' + model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # save_model_name = 'best_model.pth'
        os.makedirs(save_model_folder, exist_ok=True)
        # model_path = check_model_name(save_model_folder, save_model_name)
    elif args.save_model and args.ft:
        # model_path = './model/' + args.tags +'_train' + '/' + 'best_model.pth'
        save_model_folder = f'./results/results_FT/{args.tags}/training/' + model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # save_model_name = 'best_model.pth'
        os.makedirs(save_model_folder, exist_ok=True)
    elif args.save_model:
        # model_path = './model/' + args.tags +'_train' + '/' + 'best_model.pth'
        # save_model_folder = f'./results/{args.tags}/training/model/' 
        save_model_folder = f'./results/results_GEN/{args.tags}/training/'+ model_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # save_model_name = 'best_model.pth'
        os.makedirs(save_model_folder, exist_ok=True)
        # model_path = check_model_name(save_model_folder, save_model_name)
    else:
        # use default path
        model_path = './model/train/best_model.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # print("model folder created")
    # print("model path:", model_path)


    # ---------------------------------- #
    # Define the dataset and dataloaders #
    # ---------------------------------- #
    # if args.model == 'mobilenetv2':

    generator = torch.Generator().manual_seed(42) # fix generator for reproducible results (hopefully good)

    batch_size = args.batch_size # 32

    if args.data_aug == 'default':
        print("Using default data augmentation (thesis)")
        # Transformations for training data
        train_transform = transforms.Compose([
            # base trasnf
            # transforms.Resize((256,256)),# interpolation = transforms.InterpolationMode.BILINEAR), # might not need since randomresizedcrop extracts rand crop and then resize it to 224 dim
            # Resize(256) -> resize the image to 256*h/2, 256, in our case is the same since the images have the same width and height
            transforms.RandomResizedCrop((224,224)), #interpolation = BICUBIC), # extracts random crop from image (i.e. 300x300) and rescale it to 224x224
            transforms.RandomHorizontalFlip(), # helps the training
            # augmentations
            transforms.RandomRotation((-5,5)), # rotate the image by a random angle between -5 and 5 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # adjust the brightness, contrast, saturation and hue of the image
            # transforms.GaussianBlur((5,9), sigma=(0.1, 2.0)), # blur the image with a random kernel size and a random standard deviation
            # kernel size (5,9) taken from pytorch doc, for sigma used the default values -> https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # training and validation dataset
        train_dataset = FaceImagesDataset(train_dir, train_transform)
        train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training  -> 9600 thesis imgs in training
        val_size = len(train_dataset) - train_size # 20% of the dataset for validation -> 2400 thesis imgs in testing/validation

        # split the dataset into training and validation
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator)
        # returns two datasets: train_dataset and val_dataset

        # -------------------------------------------------------------------------------- #
        # old dataloader definition
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # -------------------------------------------------------------------------------- #

        # test with samplers for the dataloader
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val   = torch.utils.data.SequentialSampler(val_dataset)

        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train, drop_last = True, num_workers = 3) # batch_sampler = batch_sampler_train) #
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val, drop_last = False, num_workers = 3)

        # # test dataloader 
        # test_dataset = FaceImagesDataset(test_dir, test_transform)
        # sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        # test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=sampler_test) # shuffle=False)

    elif args.data_aug == 'new': 
        print("Using default data augmentation (thesis)")
        # Transformations for training data
        train_transform = transforms.Compose([
            # base trasnf
            transforms.Resize((256,256)),# interpolation = transforms.InterpolationMode.BILINEAR), # might not need since randomresizedcrop extracts rand crop and then resize it to 224 dim
            # Resize(256) -> resize the image to 256*h/2, 256, in our case is the same since the images have the same width and height
            transforms.RandomCrop((224,224)), #interpolation = BICUBIC), # extracts random crop from image (i.e. 300x300) and rescale it to 224x224
            transforms.RandomHorizontalFlip(), # helps the training
            # augmentations
            transforms.RandomRotation((-5,5)), # rotate the image by a random angle between -5 and 5 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # adjust the brightness, contrast, saturation and hue of the image
            # transforms.GaussianBlur((5,9), sigma=(0.1, 2.0)), # blur the image with a random kernel size and a random standard deviation
            # kernel size (5,9) taken from pytorch doc, for sigma used the default values -> https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # training and validation dataset
        train_dataset = FaceImagesDataset(train_dir, train_transform)
        train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training  -> 9600 thesis imgs in training
        val_size = len(train_dataset) - train_size # 20% of the dataset for validation -> 2400 thesis imgs in testing/validation

        # split the dataset into training and validation
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator)
        # returns two datasets: train_dataset and val_dataset

        # -------------------------------------------------------------------------------- #
        # old dataloader definition
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # -------------------------------------------------------------------------------- #

        # test with samplers for the dataloader
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val   = torch.utils.data.SequentialSampler(val_dataset)

        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train, drop_last = True, num_workers = 3) # batch_sampler = batch_sampler_train) #
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val, drop_last = False, num_workers = 3)


    elif args.data_aug == 'crop':
        print("Using default data augmentation (thesis)")
        # Transformations for training data
        train_transform = transforms.Compose([
            # base trasnf
            transforms.Resize((256,256)),# interpolation = transforms.InterpolationMode.BILINEAR), # might not need since randomresizedcrop extracts rand crop and then resize it to 224 dim
            # Resize(256) -> resize the image to 256*h/2, 256, in our case is the same since the images have the same width and height
            transforms.CenterCrop((224,224)), #interpolation = BICUBIC), # extracts random crop from image (i.e. 300x300) and rescale it to 224x224
            transforms.RandomHorizontalFlip(), # helps the training
            # augmentations
            transforms.RandomRotation((-5,5)), # rotate the image by a random angle between -5 and 5 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # adjust the brightness, contrast, saturation and hue of the image
            # transforms.GaussianBlur((5,9), sigma=(0.1, 2.0)), # blur the image with a random kernel size and a random standard deviation
            # kernel size (5,9) taken from pytorch doc, for sigma used the default values -> https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # training and validation dataset
        train_dataset = FaceImagesDataset(train_dir, train_transform)
        train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training  -> 9600 thesis imgs in training
        val_size = len(train_dataset) - train_size # 20% of the dataset for validation -> 2400 thesis imgs in testing/validation

        # split the dataset into training and validation
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator)
        # returns two datasets: train_dataset and val_dataset

        # -------------------------------------------------------------------------------- #
        # old dataloader definition
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # -------------------------------------------------------------------------------- #

        # test with samplers for the dataloader
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val   = torch.utils.data.SequentialSampler(val_dataset)

        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train, drop_last = True, num_workers = 3) # batch_sampler = batch_sampler_train) #
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val, drop_last = False, num_workers = 3)


    else:
        print("Data augmentation not supported")
        exit()

    print("data augmentations: ", args.data_aug)
    print(train_transform)
    # breakpoint()
    # -------------------------------------------------------- #
    # Define the dataset and dataloaders
    # NOTE: to be updated to load GOTCHA dataset as well

    # -------------------------------------------------------- #
    # Train the model
    # -------------------------------------------------------- #
    # training = args.training # True or False
    # print("training", training)
    # ()
    # save_model_path = args.save_model_path

    print(f"Training the model {model_name}...")

    if args.wandb:
        run.tags = run.tags + ('Training',) # add testing tag

    # # creating a folder where to save the testing results 
    if args.tl:
        exp_results_path = f'./results/results_TL/{args.tags}/training/' # i.e. ./results/EfficientNetB4_FF_no_occ_focal_loss/testing
    elif args.ft: 
        exp_results_path = f'./results/results_FT/{args.tags}/training'
    elif args.gen:
        exp_results_path = f'./results/results_GEN/{args.tags}/training'
    else: 
        exp_results_path = f'./results/results/{args.tags}/training'

    if args.save_log:
        print(f"logging results - {args.tags}")
        # set up logger
        log_path = exp_results_path + '/exp_log/output.log'
        log = create_logger(log_path)
        log.info(f"Training the {args.tags} model on the {args.dataset} dataset with threshold {args.thresh}")

        # print some info on the model architecture
        log.info("Model informations:")
        log.info(f"Model Name: {model_name}")
        log.info(f"Pretrained model weights: {pretrained_model_path if pretrained_model_path != '' else 'ImageNet pre-trained model weights'}")
        log.info(f"Optimizer: {args.optimizer}")
        log.info(f"Loss function: {criterion}")
        log.info(f"Learning rate: {args.lr}")
        log.info(f"Batch size: {batch_size}")
        log.info(f"Number of epochs: {num_epochs}")
        # log.info(f"Resume training from epoch: ", checkpoint['epoch']+1) if args.resume else print("training model from scratch")
        log.info(f"Early Stopping-Patience: {patience}")
        log.info(f"Dataset: {args.dataset}")
        log.info(f"Dataset path: {train_dir}")
        log.info(f"Data augmentation: {args.data_aug}")
        log.info(f"{train_transform}")
        # log.info(f"Model path: {model_path}")
        log.info(f"Training the model...")
        

    # gradcam_save_path = ""

    start_epoch = 0
    if args.resume:
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and 'criterion' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1 
            criterion = checkpoint['criterion']
        log.info(f"Resume training from epoch: {start_epoch}")
    
    else:
        print("Training the model from scratch (epoch 0).")
        # log.info("Training the model from scratch (epoch 0)")
    
    # ()

    # Train the model
    for epoch in range(start_epoch, args.num_epochs):
        
        if not gotcha:
            train_loss, train_accuracy, train_accuracy_original, train_accuracy_simswap, train_accuracy_ghost, train_accuracy_facedancer = train_one_epoch(model, criterion, optimizer, train_dataloader, device, args.thresh)
            
            # lr_scheduler.step() # update the learning rate based on the validation loss

            val_loss, val_accuracy, val_accuracy_original, val_accuracy_simswap, val_accuracy_ghost, val_accuracy_facedancer, balanced_test_acc, TPR, TNR, auc, eer, ap_score = validate_one_epoch(model, criterion, val_dataloader, device, exp_results_path, epoch, args.thresh) #, thresh = 0.6)
            
            # print train and validation info
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
            print('--------------------------------------------------------------------')
            # print some more tain adn val info for each method
            print(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy SimSwap: {train_accuracy_simswap:.4f}, Train Accuracy Ghost: {train_accuracy_ghost:.4f} Train Accuracy FaceDancer: {train_accuracy_facedancer:.4f}")
            print(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy SimSwap: {val_accuracy_simswap:.4f}, Validation Accuracy Ghost: {val_accuracy_ghost:.4f}, Validation Accuracy FaceDancer: {val_accuracy_facedancer:.4f}")
            print(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc:.4f}, EER: {eer:.4f}")
            print('--------------------------------------------------------------------')

            # log all train and vall info to wandb
            if args.wandb:
                run.log({'Train Loss': train_loss, 'Train Accuracy': train_accuracy})
                run.log({'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})
                run.log({'Epoch': epoch+1})
                run.log({'Train Accuracy Original': train_accuracy_original, 'Train Accuracy SimSwap': train_accuracy_simswap, 'Train Accuracy Ghost': train_accuracy_ghost, 'Train Accuracy FaceDancer': train_accuracy_facedancer, 'Validation Accuracy FaceDancer': val_accuracy_facedancer})
                run.log({'Validation Accuracy Original': val_accuracy_original, 'Validation Accuracy SimSwap': val_accuracy_simswap, 'Validation Accuracy Ghost': val_accuracy_ghost})
            
            if args.save_log:
                log.info(f'Epoch: {epoch+1}/{num_epochs}:'),
                log.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
                log.info(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy SimSwap: {train_accuracy_simswap:.4f}, Train Accuracy Ghost: {train_accuracy_ghost:.4f} Train Accuracy FaceDancer: {train_accuracy_facedancer:.4f}")
                log.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
                log.info(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy SimSwap: {val_accuracy_simswap:.4f}, Validation Accuracy Ghost: {val_accuracy_ghost:.4f}, Validation Accuracy FaceDancer: {val_accuracy_facedancer:.4f}")
                log.info(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc:.4f}, EER: {eer:.4f}")
                log.info('--------------------------------------------------------------------')
            

            # save checkpoint at each epoch
            # checkpoint_paths = [save_model_folder + '/checkpoint.pth' ]

            # At the end of each epoch, check the validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"best val_loss: {best_val_loss:.4f}")
                print('--------------------------------------------------------------------')
                early_stopping_counter = 0

                # Save the best model
                # checkpoint_paths.append(save_model_folder + '/best_checkpoint.pth')
                best_model_path = save_model_folder + '/best_checkpoint.pth'
                # -------------------------------------------------- #
                # save all model info
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'criterion': criterion,
                    'optimizer': optimizer.state_dict(),
                    # 'train_loss': train_loss,
                    # 'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'args': args
                }, best_model_path)
                print(f"best checkpoint saved in {best_model_path}")
                log.info(f"Best checkpoint saved in: {best_model_path}")
                log.info('--------------------------------------------------------------------')
                # -------------------------------------------------- #
                
            else:
                
                # update early stopping condition
                early_stopping_counter += 1
                print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                print('--------------------------------------------------------------------')
                log.info(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                log.info('--------------------------------------------------------------------')
                if early_stopping_counter >= patience:
                    print('Early stopping')
                    log.info("Early Stopping")
                    log.info('--------------------------------------------------------------------')
                    break
            
            
        # ------- #
        # GOTCHA! #
        # ------- #
        else:

            train_loss, train_accuracy, train_accuracy_original, train_accuracy_dfl, train_accuracy_fsgan = gotcha_train_one_epoch(model, criterion, optimizer, train_dataloader, device, args.thresh)
            
            # lr_scheduler.step() # update the learning rate based on the validation loss

            val_loss, val_accuracy, balanced_test_acc, val_accuracy_original, val_accuracy_dfl, val_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer = gotcha_validate(model, criterion, val_dataloader, device, args.thresh) #, thresh = 0.6)
            
            # print train and validation info
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
            print('--------------------------------------------------------------------')
            # print some more tain adn val info for each method
            print(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy DFL: {train_accuracy_dfl:.4f}, Train Accuracy FSGAN: {train_accuracy_fsgan:.4f}")
            print(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy DFL: {val_accuracy_dfl:.4f}, Validation Accuracy FSGAN: {val_accuracy_fsgan:.4f}")
            print(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc_score:.4f}, EER: {eer:.4f}")
            print('--------------------------------------------------------------------')

            # log all train and vall info to wandb
            if args.wandb:
                run.log({'Train Loss': train_loss, 'Train Accuracy': train_accuracy})
                run.log({'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})
                run.log({'Epoch': epoch+1})
                run.log({'Train Accuracy Original': train_accuracy_original, 'Train Accuracy DFL': train_accuracy_dfl, 'Train Accuracy FSGAN': train_accuracy_fsgan})
                run.log({'Validation Accuracy Original': val_accuracy_original, 'Validation Accuracy DFL': val_accuracy_dfl, 'Validation Accuracy FSGAN': val_accuracy_fsgan})
            
            if args.save_log:
                log.info(f'Epoch: {epoch+1}/{num_epochs}:'),
                log.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}")
                log.info(f"Train Accuracy Original: {train_accuracy_original:.4f}, Train Accuracy DFL: {train_accuracy_dfl:.4f}, Train Accuracy FSGAN: {train_accuracy_fsgan:.4f}")
                log.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
                log.info(f"Validation Accuracy Original: {val_accuracy_original:.4f}, Validation Accuracy DFL: {val_accuracy_dfl:.4f}, Validation Accuracy FSGAN: {val_accuracy_fsgan:.4f}")
                log.info(f"Balanced Accuracy: {balanced_test_acc:.4f}, AP: {ap_score:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}, AUC: {auc_score:.4f}, EER: {eer:.4f}")
                log.info('--------------------------------------------------------------------')
                        
            # save checkpoint at each epoch
            checkpoint_paths = [save_model_folder + '/checkpoint.pth' ] 

            # At the end of each epoch, check the validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"best val_loss: {best_val_loss:.4f}")
                print('--------------------------------------------------------------------')
                early_stopping_counter = 0

                # Save the best model
                # checkpoint_paths.append(save_model_folder + '/best_checkpoint.pth')
                best_model_path = save_model_folder + '/best_checkpoint.pth'
                # save all model info
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'criterion': criterion,
                    'optimizer': optimizer.state_dict(),
                    # 'train_loss': train_loss,
                    # 'train_accuracy': train_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'args': args
                }, best_model_path)
                print(f"best checkpoint saved in: {best_model_path}")
                log.info(f"Best checkpoint saved in: {best_model_path}")
                log.info('--------------------------------------------------------------------')
                
            else:
                early_stopping_counter += 1
                print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                print('--------------------------------------------------------------------')
                log.info(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                log.info('--------------------------------------------------------------------')
                if early_stopping_counter >= patience:
                    print('Early stopping')
                    log.info("Early Stopping")
                    log.info('--------------------------------------------------------------------')
                    break
    # close wandb session
    if args.wandb:
        run.finish() # end the wandb session

if __name__ == '__main__':
    main()