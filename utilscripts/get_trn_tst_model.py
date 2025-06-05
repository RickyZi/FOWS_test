# given the args from the command line, get the model path
import os
import sys
import argparse
import torch
# import torch.nn as nn
import torch.models as models
import torch.nn as nn
import timm  # for xception model
from torchvision import models


# ------------------------------------------------------------------------------------------- #
# testing models prompt
# python test_baseline.py --model "effnetb4_thesis_no_occ" --dataset "gotcha_no_occ"  --ft --tags "EffNetB4_thesis_no_occ_FT_vs_gotcha_no_occ"
# model_ effnetb4_baseline
# training_dataset: fows_no_occ
# testing_dataset: gotcha_no_occ

# @TODO: extract model name from --model argument (1st part), and dataset name from --dataset argument (2nd part) -> add also FT/TL argument info to the model_info string
# combine them to create the model_info string
# retrieve the best_model.pth file from the model_weights folder 
# -> find the folder with the same model name and dataset name as in the model_info string 
# return the path to the best_model.pth file


def get_model_path(model_name, dataset, trn_strategy, models_folder = '../model_weights/'):

    # NOTE: the defaults models_folder is the folder of the pretrained baselines.
    # Change the models_folder to be the one where you save the model during training

    # Check if the model is valid
    if model_name.lower() not in ['mnetv2', 'effnetb4', 'xception']:
        print(f"Model {model_name} is not supported")
        exit()
    # Check if the dataset is valid
    if dataset.lower() not in ['fows_occ', 'fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']: 
        print(f"Dataset {dataset} is not valid")
        exit()
    
    # Get the model path based on the arguments
    model_info = model_name + '_' + dataset + '_' + trn_strategy.upper() #('_FT' if ft else '_TL')
    # '_TL' if tl else 
    print("Model info: ", model_info)
    
    # models_folder = '../model_weights/' + trn_strategy.upper() + '/'
    models_folder += trn_strategy.upper() + '/' 

    pretrained_model_path = ''

    # Check if the model path exists
    if not os.path.exists(models_folder):
        print(f"Model path {models_folder} does not exist")
        exit()
    
    # navigate all model folders subdirectories and check if the model_info is in any of them
    found = False
    for root, dirs, files in os.walk(models_folder):
        for dir in dirs:
            # print(f"Checking directory: {dir}")
            if dir.lower() == model_name.lower():  # check if the directory name is the same as the model_info
                # find the .pth file in the directory
                print(f"Found directory: {dir} with same name as {model_info}")
                model_dir = os.path.join(root, dir)
                for sub_root, sub_dirs, sub_files in os.walk(model_dir):
                    for file in sub_files:
                        if file.endswith('.pth') and model_info.lower() in file.lower():
                            found = True
                            pretrained_model_path = os.path.join(sub_root, file)
                            # print(f"Pretrained model path: {pretrained_model_path}")
                            break
                break # uncomment this if you want to stop searching after finding the first directory
                
    if not found:
        print(f"NO model {model_info} found in {models_folder}")
        exit()
    
    return pretrained_model_path
# ------------------------------------------------------------------------------------------- #
# def get_model_path(model_weights_path, model_str):
#     model_path = ''
#     for root, dirs, files in os.walk(model_weights_path):
#         for file in files:
#             if file.endswith('.pth') and model_str in file:
#                 print(os.path.join(root, file))
#                 model_path = os.path.join(root, file)
#                 found = True
#     if found:
#         print(model_path)
#     else:
#         print("no model found")
#         exit()

#     return model_path

# ------------------------------------------------------------------------------------------- #
# def get_pretrained_model(args):
def load_model_from_path(args):
    # model_str = model_name + '_' + dataset + '_' + trn_strategy
    trn_strategy = 'TL' if args.tl else 'FT'  
    model_name = args.model
    trn_dataset = args.trn_dataset
    # model_str = args.model + '_' + args.train_dataset + '_' + 'FT' if args.ft else 'TL'
    # if model_name == 'mnetv2': 
    if args.model.lower() == 'mnetv2':
        print("Loading MobileNetV2 model")
        # model = mobilenet_v2(pretrained=True)
        model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2') 
        model_name = 'MobileNetV2' # add the model name to the model object

        if args.tl:
            # ---------------------------- #
            # --- TRANSFER LEARNING!!! --- #
            # ---------------------------- #
            print("loading pretrained TL model")
            # # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

            if args.train_dataset == 'gotcha_no_occ':
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                print("model saved in:", pretrained_model_path)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif args.train_dataset in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                print("model saved in:", pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))
            else:
                print("no pretrained model found")
                exit()

            print("model saved in:", pretrained_model_path)
            print("model loaded!")

        elif args.ft:
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            print("loading pretrained FT model")
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) 
            
            if args.train_dataset in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                print("model saved in:", pretrained_model_path)
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            else:
                print("no pretrained model found")
                exit()

        else: 
            print("using pretrained ImageNet model")
            pretrained_model_path = 'MobileNetV2 Pre-trained ImageNet Model'
            # 
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
    
    elif args.model.lower() == 'effnetb4':
        print("Loading EfficientNetB4 model")
        # breakpoint()
        # pip install efficientnet_pytorch
        # run this command to install the efficientnet model
        # model = EfficientNet.from_pretrained('efficientnet-b4')

        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
        # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html
        model_name = 'EfficientNet_B4' # add the model name to the model object

        # load the pre-trained model for testing (load model weights)
        if args.tl:
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
            
            if args.train_dataset == 'gotcha_no_occ':
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                # model.load_state_dict(torch.load(pretrained_model_path))
                print("model saved in:", pretrained_model_path)
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif args.train_dataset in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                print("model saved in:", pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))
            else:
                print("no pretrained model found")
                exit()

        elif args.ft:
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            print("loading pretrained FT model")
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            # if args.dataset == 'thesis_occ':
            if args.train_dataset in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
                # pretrained_model_path = get_model_path(model_name, dataset, tl, ft)
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                print("model saved in:", pretrained_model_path)
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
        else:
            print("using pretrained ImageNet model")
            pretrained_model_path = 'EfficientNet_B4 Pre-trained ImageNet Model'
            # Replace the classifier layer
            # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
            # check_model_gradient(model)
        
    # elif model_name == 'xception':
    elif args.model.lower() == 'xception':
        print("Loading pretrained XceptionNet model...")
        # load the xceptionet model
        # pip install timm
        # import timm
        model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
        model_name = 'XceptionNet' # add the model name to the model object
        # load the pre-trained model for testing (load model weights)
        # pretrained_model_path = args.save_model_path 
        if args.tl:
            # ---------------------------- #
            # --- TRANSFER LEARNING!!! --- #
            # ---------------------------- #
            print("loading pretrained model")

            if args.train_dataset in ['gotcha_occ', 'gotcha_no_occ']:
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif 'fows_occ' in args.train_dataset or 'fows_no_occ' in args.train_dataset:
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                print("model saved in:", pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))

            else:
                print("no pretrained model found")
                exit()

        elif args.ft:
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            if args.train_dataset in ['fows_occ, fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']:
                print("loading pretrained FT model")
                # pretrained_model_path = get_model_path(model_name, dataset, tl, ft)
                pretrained_model_path = get_model_path(model_name, trn_dataset, trn_strategy, args.model_path)
                # ------------------------------------------ #
                # test with Adam optimizer
                # pretrained_model_path = './results/results_FT/Xception_gotcha_no_occ_ADAM_FT/training/XceptionNet_2025-03-06-11-51-53/best_checkpoint.pth'
                # ------------------------------------------ #
                print("model saved in:", pretrained_model_path)
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            else:
                print("no pretrained model found")
                exit()

        else:
            print("using pretrained ImageNet model")
            pretrained_model_path = 'Xception Pre-trained ImageNet Model'

        # print("Model loaded!")
        # model.to(device)
        # print(model)

    else:
        print("Model not supported")
        exit()
    
    return model, model_name

# -------------------------------------------------------------------------------------------- #

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
            model_name = 'MobileNetV2_TL' # add the model name to the model object
        
        elif args.ft:
            print("Fine-Tuning MobileNetV2 pre-trained model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            model_name = 'MobileNetV2_FT' # add the model name to the model object
            print("Transfer learning - Freezing all layers except the classifier")
            # # Freeze all layers
            # for param in model.parameters():
            #     param.requires_grad = False
            
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face
            # ()
            

        else:
            # print("problem in loading the model")
            # exit()
            print("Loading MobileNetV2 pre-trained model")
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V2')
            model_name = 'MobileNetV2' # add the model name to the model object
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

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
        # ------------------------------------------------------------------------ #
        elif args.ft: 
            print("Fine-Tuning the EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real or swap face
            model_name = 'EfficientNet_B4_FT'
        else: 
            # print("problem in loading the model")
            # exit()
            print("Loading EfficientNetB4 pre-trained model")
            model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            # model.to(device)
            model_name = 'EfficientNet_B4' # add the model name to the model object
       
        
    elif args.model == 'xception':
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

        elif args.ft: 
            print("Fine Tuning XceptionNet pre-trained model")
            # print("Transfer learning - Freezing all layers except the classifier")
            model = timm.create_model('xception', pretrained=True) #, num_classes=1) # only 1 output -> prob of real or swap face
            model_name = 'XceptionNet' # add the model name to the model object
            
            # Replace the classifier layer
            in_features = model.get_classifier().in_features
            model.fc = nn.Linear(in_features, 1)  # modify the last layer of the classifier to have 1 output -> prob of real or swap face

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
           
            
        
    else:
        print("Model not supported")
        exit()


    return model, model_name

# -------------------------------------------------------------------------------------------- #
# # Create the argument parser (reduced version)
# def get_args_parse():
#     parser = argparse.ArgumentParser(description='Model Training and Testing')
#     # Add arguments

#     # model parameters
#     parser.add_argument('--model', type=str, default='mnetv2', help='Model to use for training and testing')
    
#     # parser.add_argument('--tl', action = 'store_true', help='Use the re-trained version of the model (transf learning)')
#     parser.add_argument('--ft', action = 'store_false', help='Fine-Tuning the model') # store_false -> default is True
#     # dataset parameters
#     parser.add_argument('--dataset', type=str, default='fows_occ', help='Path to the training and/or testing dataset')
#     return parser

# # ------------------------------------------------------------------------------------------- #

# def main():
#     # Get the arguments from the command line
#     parser = get_args_parse()
#     args = parser.parse_args()

#     print(args)

#     pretrained_model_path = get_model_path(args)    
#     print("Pretrained model path: ", pretrained_model_path)

#     # load the model to gpu if available
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("device:", device)

#     # load the model


#     # models: mnetv2, effnetb4, xception, icpr2020
#     # datasets: fows_occ, fows_no_occ, gotcha_occ, gotcha_no_occ
#     # tl: transfer learning, trn_strategy == 'ft' fine tuning -> NOTE: only FT is available for now

    
#     # except Exception as e:
#     #     print(f"Error while searching for the model: {e}")
#     #     sys.exit(1)


#     # load the model from the path and print it
#     # model_path = get_model_path(args)
#     # try:
#     #     model = torch.load(os.path.join(root, file))
#     #     print(model)
#     # except Exception as e:
#     #     # print(f"Error while loading the model: {e}")
#     #     # sys.exit(1)
#     #     best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
#     #     model.load_state_dict(best_ckpt['model'])

#     # best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
#     # model.load_state_dict(best_ckpt['model'])
#     # print(model)


# if __name__ == '__main__':
#     main()
