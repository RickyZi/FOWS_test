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


def get_pretrained_path(model_name, dataset, tl, ft):
    # Check if the model is valid
    if model_name not in ['mnetv2', 'effnetb4', 'xception', 'icpr2020', 'neurips2023']:
        print(f"Model {model_name} is not supported")
        sys.exit(1)
    # Check if the dataset is valid
    if dataset not in ['fows_occ', 'fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']: 
        print(f"Dataset {dataset} is not valid")
        sys.exit(1)
    
    # Get the model path based on the arguments
    model_info = model_name + '_' + dataset + ('_FT' if ft else '_TL')
    # '_TL' if tl else 
    print("Model info: ", model_info)

    if tl:
        models_folder = 'model_weights/TL/'
    else:
        models_folder = 'model_weights/FT/' # to be updated to 'model_weights/FT/' if FT is used
    pretrained_model_path = ''

    # Check if the model path exists
    if not os.path.exists(models_folder):
        print(f"Model path {models_folder} does not exist")
        sys.exit(1)
    
    # navigate all model folders subdirectories and check if the model_info is in any of them
    found = False
    for root, dirs, files in os.walk(models_folder):
        for dir in dirs:
            # print(f"Checking directory: {dir}")
            if dir.lower() == model_info.lower():  # check if the directory name is the same as the model_info
                # find the .pth file in the directory
                print(f"Found directory: {dir} with same name as {model_info}")
                model_dir = os.path.join(root, dir)
                for sub_root, sub_dirs, sub_files in os.walk(model_dir):
                    for file in sub_files:
                        if file.endswith('.pth'):
                            found = True
                            pretrained_model_path = os.path.join(sub_root, file)
                            # print(f"Pretrained model path: {pretrained_model_path}")
                            break
                break # uncomment this if you want to stop searching after finding the first directory
                
    if not found:
        print(f"NO model {model_info} found in {models_folder}")
        sys.exit(1)

    
    return pretrained_model_path

# ------------------------------------------------------------------------------------------- #
def load_model_from_path(model_name, device):
    # if model_name == 'mnetv2': 
    if 'mnetv2' in model_name.lower():
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
            for param in model.parameters():
                param.requires_grad = False
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

            pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)

            # best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")

            # if 'model' in best_ckpt.keys():
            #     model.load_state_dict(best_ckpt['model'])
            # else:
            #     model.load_state_dict(best_ckpt)

            if 'gotcha_no_occ' in model_name:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif model_name in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                # print("model saved in:", pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))
            else:
                print("no pretrained model found")
                exit()

            print("model saved in:", pretrained_model_path)
            print("model loaded!")

        elif trn_strategy == 'ft':
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            print("loading pretrained FT model")
            # Replace the classifier layer
            model.classifier[1] = nn.Linear(model.last_channel, 1) 
            
            if model_name in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
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
            # check_model_gradient(model)
            # 

        model.to(device)
        print("Model loaded!")
        # print(model)
        print("model_name: ", model_name)
        print("args.dataset: ", dataset)
        print("model_path: ", pretrained_model_path)
        # exit()
        # 
        # print(model)
        # check_model_gradient(model)
        # breakpoint()


    # elif model_name == 'effnetb4':
    elif 'effnetb4' in model_name.lower():
        print("Loading EfficientNetB4 model")
        # breakpoint()
        # pip install efficientnet_pytorch
        # run this command to install the efficientnet model
        # model = EfficientNet.from_pretrained('efficientnet-b4')

        model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
        # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html
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
            
            if 'gotcha_no_occ' in model_name:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif model_name in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                # print("model saved in:", pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))
            else:
                print("no pretrained model found")
                exit()

        elif trn_strategy == 'ft':
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            print("loading pretrained FT model")
            # Replace the classifier layer
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            # if args.dataset == 'thesis_occ':
            if model_name in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
        else:
            print("using pretrained ImageNet model")
            pretrained_model_path = 'EfficientNet_B4 Pre-trained ImageNet Model'
            # Replace the classifier layer
            # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
            # check_model_gradient(model)
            # 
        # pretrained_model_path = args.save_model_path 
        
        model.to(device)
        print("Model loaded!")

        # for key in model.keys():
        #     print(key)

        # print(model)
        # exit()
        
    # elif model_name == 'xception':
    elif 'xception' in model_name.lower():
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

            if 'gotcha_no_occ' in model_name or 'gotcha_occ' in model_name:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif 'fows_occ' in model_name or 'fows_no_occ' in model_name:
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
                # print("model saved in:", pretrained_model_path)
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
                pretrained_model_path = get_pretrained_path(model_name, dataset, tl, ft)
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

    else:
        print("Model not supported")
        exit()
    
    return model, pretrained_model_path

# --------------------------------------------------------------------------------------------#
# def load_model_from_path(args, device):
    # if args.model == 'mnetv2': 
    if 'mnetv2' in args.model.lower():
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

            if 'gotcha_no_occ' in args.model:
                pretrained_model_path = get_pretrained_path(args)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif args.model in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
                pretrained_model_path = get_pretrained_path(args)
                # print("model saved in:", pretrained_model_path)
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
            
            if args.model in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
                pretrained_model_path = get_pretrained_path(args)
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
            # check_model_gradient(model)
            # 

        model.to(device)
        print("Model loaded!")
        # print(model)
        print("args.model: ", args.model)
        print("args.dataset: ", args.dataset)
        print("model_path: ", pretrained_model_path)
        # exit()
        # 
        # print(model)
        # check_model_gradient(model)
        # breakpoint()


    # elif args.model == 'effnetb4':
    elif 'effnetb4' in args.model.lower():
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
            
            if 'gotcha_no_occ' in args.model:
                pretrained_model_path = get_pretrained_path(args)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif args.model in ['fows_occ', 'fows_no_occ', 'gotcha_occ']:
                pretrained_model_path = get_pretrained_path(args)
                # print("model saved in:", pretrained_model_path)
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
            if args.model in ['fows_occ', 'gotcha_occ', 'fows_no_occ', 'gotcha_no_occ']:
                pretrained_model_path = get_pretrained_path(args)
                
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
        else:
            print("using pretrained ImageNet model")
            pretrained_model_path = 'EfficientNet_B4 Pre-trained ImageNet Model'
            # Replace the classifier layer
            # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
            # check_model_gradient(model)
            # 
        # pretrained_model_path = args.save_model_path 
        
        model.to(device)
        print("Model loaded!")

        # for key in model.keys():
        #     print(key)

        # print(model)
        # exit()
        
    # elif args.model == 'xception':
    elif 'xception' in args.model.lower():
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

            if 'gotcha_no_occ' in args.model or 'gotcha_occ' in args.model:
                pretrained_model_path = get_pretrained_path(args)
                # model.load_state_dict(torch.load(pretrained_model_path))
                best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
                model.load_state_dict(best_ckpt['model'])
            elif 'fows_occ' in args.model or 'fows_no_occ' in args.model:
                pretrained_model_path = get_pretrained_path(args)
                # print("model saved in:", pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))

            else:
                print("no pretrained model found")
                exit()

        elif args.ft:
            # ---------------------- #
            # --- FINE TUNING!!! --- #
            # ---------------------- #
            if args.model in ['fows_occ, fows_no_occ', 'gotcha_occ', 'gotcha_no_occ']:
                print("loading pretrained FT model")
                pretrained_model_path = get_pretrained_path(args)
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
            # check_model_gradient(model)
            # 
            
        
        # model.to(device)
        # print("Model loaded!")

        # print(model)
        # check_model_gradient(model)
        # 

    else:
        print("Model not supported")
        exit()
    
    return model, pretrained_model_path
# ------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------- #
# Create the argument parser (reduced version)
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # Add arguments

    # model parameters
    parser.add_argument('--model', type=str, default='mnetv2', help='Model to use for training and testing')
    
    # parser.add_argument('--tl', action = 'store_true', help='Use the re-trained version of the model (transf learning)')
    parser.add_argument('--ft', action = 'store_false', help='Fine-Tuning the model') # store_false -> default is True
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='fows_occ', help='Path to the training and/or testing dataset')
    return parser

# ------------------------------------------------------------------------------------------- #

def main():
    # Get the arguments from the command line
    parser = get_args_parse()
    args = parser.parse_args()

    print(args)

    pretrained_model_path = get_model_path(args)    
    print("Pretrained model path: ", pretrained_model_path)

    # load the model to gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load the model


    # models: mnetv2, effnetb4, xception, icpr2020
    # datasets: fows_occ, fows_no_occ, gotcha_occ, gotcha_no_occ
    # tl: transfer learning, trn_strategy == 'ft' fine tuning -> NOTE: only FT is available for now

    
    # except Exception as e:
    #     print(f"Error while searching for the model: {e}")
    #     sys.exit(1)


    # load the model from the path and print it
    # model_path = get_model_path(args)
    # try:
    #     model = torch.load(os.path.join(root, file))
    #     print(model)
    # except Exception as e:
    #     # print(f"Error while loading the model: {e}")
    #     # sys.exit(1)
    #     best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    #     model.load_state_dict(best_ckpt['model'])

    # best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
    # model.load_state_dict(best_ckpt['model'])
    # print(model)


if __name__ == '__main__':
    main()
