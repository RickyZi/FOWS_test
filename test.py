# ------------------------------------------------------------ #
# code for testing the pretrained models 
# ------------------------------------------------------------ #

"""
ideally it should be a "summary" of all training codes for the models
    - mnetv2
    - effnetb4
    - xception
    - icpr2020 (Milan_EffNetB4)
    - Neurips2023 (DFB_xceptionNet)

# ------------------------------------------------------------------- #
NOTE:
    - load only the model weights we decide to publish (TL and FT)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os
from customDataset import FaceImagesDataset
from utilscripts.papers_data_augmentations import *
from utilscripts.logger import create_logger
from utilscripts.customDataset import FaceImagesDataset
from utilscripts.gotcha_dataset import gotcha_test
from model_training_scripts.bceWLL_test.utils.train_val_test import test_one_epoch
import utilscripts.fornet as fornet
from utilscripts.fornet import *
from focalLoss import FocalLoss
# import wandb # for logging results to wandb
import argparse # for command line arguments
# import timm # for using the XceptionNet model (pretrained)
# pip install timm # install the timm package to use the XceptionNet model
# import yaml
import random
from model_training_scripts.bceWLL_test.utils.get_pretrained_model import get_pretrained_model_path

# ------------------------------------------------------------------------------------------- #
# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # Add arguments

    # model parameters
    parser.add_argument('--model', type=str, default='mnetv2', help='Model to use for training and testing')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--thresh', type = float, default = 0.5, help = 'Threshold for binary classification.')
    # add flag to save logs to a file in the model folder
    parser.add_argument('--save-log', action='store_false', help='Save the model output logs')
    # path to the model folder -> to be updated based on the model name and tags
    parser.add_argument('--save-model-path', action= "store_false", help='Use the saved pre-trained version of the model')
    parser.add_argument('--tl', action = 'store_true', help='Use the re-trained version of the model (transf learning)')
    parser.add_argument('--ft', action = 'store_true', help='Fine-Tuning the model')
    # parser.add_argument('--gen', action = 'store_true', help='Testing the generalization capabilites of the model')
    # parser.add_argument('--robust', action = 'store_true', help='Testing the robusteness of the model to different data augmentations')
    parser.add_argument('--data-aug', type=str, default='default', help='Data augmentation to use for training and testing') 
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='thesis_occ', help='Path to the training and/or testing dataset')
    # add wandb argument
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
    # ------------------------------- #
    init_seed()
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

    print("args.model: ", args.model)
    print("args.dataset: ", args.dataset)
    print("args.data_aug: ", args.data_aug)
    

    pretrained_model_path = get_pretrained_model_path(args, model)

    print("model_path: ", pretrained_model_path)
    
    # load the model to the GPU
    # load the model
    if args.model == 'icpr2020':
        net_name = "EfficientNetB4"
        net_class = getattr(fornet, net_name)
        model: FeatureExtractor = net_class().to(device)
        model_state = torch.load(pretrained_model_path, map_location = "cpu")
        incomp_keys = model.load_state_dict(model_state['net'], strict=True)
        print(incomp_keys)
        print(model)
        print('Model loaded!')
        args.dataset = 'milan_occ' if args.dataset == 'thesis_occ' else 'milan_no_occ' # replace the dataset name with the one used in the training
        # args.data_aug = 'milan'
        model_name = args.model
    elif args.model == 'neurips2023':
        # net_name = "DFB_xceptionNet"
        # @TODO: check how to load the DFB model and test it 
        print("Neurips2023 model is not implemented yet!")
        exit()
    else:
        print("model saved in:", pretrained_model_path)
        model_name = args.model
        # model.load_state_dict(torch.load(pretrained_model_path))
        best_ckpt = torch.load(pretrained_model_path, map_location = "cpu")
        model.load_state_dict(best_ckpt['model']) 
        model.to(device)
        print(model)
        print("Model loaded!")

    print("args.dataset: ", args.dataset)
    breakpoint()

    # ---------------------------------- #
    # Define the dataset and dataloaders #
    gotcha = False
    if args.dataset == 'fows_occ':
        if args.model=='icpr2020':
            test_dir = '/media/data/rz_dataset/milan_faces/occlusion/testing/'
            args.data_aug = 'icpr2020'
        else: 
            test_dir = '/media/data/rz_dataset/users_face_occlusion/testing/'
        
    elif args.dataset == 'fows_no_occ':
        if args.model=='icpr2020':
            test_dir = '/media/data/rz_dataset/milan_faces/no_occlusion/testing/'
            args.data_aug = 'icpr2020'
        else: 
            test_dir = '/media/data/rz_dataset/user_faces_no_occlusion/testing/'
        
    elif args.dataset == 'gotcha_occ':
        test_dir = '/media/data/rz_dataset/gotcha/balanced_gotcha/occlusion/testing'
        gotcha = True

    elif args.dataset == 'gotcha_no_occ':
        test_dir = '/media/data/rz_dataset/gotcha/balanced_gotcha/no_occlusion/testing'
        gotcha = True

    else:
        print("Dataset not supported")
        exit()

    # ---------------------------------- #
    print("dataset:", args.dataset)
    print("test_dir:", test_dir)
    print("batch_size: ", args.batch_size)
    print("data_aug: ", args.data_aug)

    # ---------------------------------- #
    # Define the dataset and dataloaders #
    # ---------------------------------- #
    if args.data_aug:
        print("Using default data augmentation")

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
        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size,  
                                    sampler=sampler_test,
                                    num_workers = 3)
    elif args.data_aug == 'icpr2020':
        print("using data augmentation from Icpr2020 paper")
        test_transform = milan_test_transf()
        # test dataloader 
        test_dataset = FaceImagesDataset(dataset_dir=test_dir, transform=test_transform, use_albu=True, training = False) #, challenge=args.trn_challenge, algo=args.trn_algo)
        # print("test_dataset: ", len(test_dataset)) # 4800 thesis_occ/thesis_no_occ
        dataset_labes = test_dataset.label_map
        print(dataset_labes)

        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size,
                                    sampler=sampler_test, 
                                    num_workers = 3)
    
    elif args.data_aug == 'neurips2023':
        print("using data augmentation from Neurips2023 paper")
        # test_transform

    else:
        print("Data augmentation not supported")
        exit()

    print("data augmentations: ", args.data_aug)
    print(test_transform)

    # -------------------------------------------------------- #
    # Testing the model
    # -------------------------------------------------------- #

    # print(f'Testing the pretrained model {model_name} on the {args.dataset} dataset...')
    print(f'Testing the pretrained model {args.model} on the {args.dataset} dataset...')

    # creating a folder where to save the testing results 
    if args.tl:
            exp_results_path = f'./results/results_TL/{args.tags}/testing'
    elif args.ft:
            exp_results_path = f'./results/results_FT/{args.tags}/testing'
    elif args.gen:
        exp_results_path = f'./results/results_GEN/{args.tags}/testing'
    else: 
        exp_results_path = f'./results/results/{args.tags}/testing'

    # Evaluate the model on the test data
    if not gotcha:
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

        # # log results to wandb
        # if args.wandb:
        #     run.tags = run.tags + ('Testing',) # add testing tag
            
        #     run.log({'Test Accuracy': test_accuracy})
        #     run.log({'Balanced Test Accuracy': balanced_test_acc})
        #     run.log({'Test Accuracy Original': test_accuracy_original}) 
        #     run.log({'Test Accuracy DFL': test_accuracy_dfl}) 
        #     run.log({'Test Accuracy FSGAN': test_accuracy_fsgan})
            
        #     run.log({'AP': ap_score})
        #     run.log({'TPR': TPR})
        #     run.log({'TNR': TNR})
        #     run.log({'AUC': auc_score})
        #     run.log({'EER': eer})
            

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
            log.info("Note the labels/preds/imgs_paths are in the json file")
            log.info("")

    # # close wandb session
    # if args.wandb:
    #     run.finish() # end the wandb session

if __name__ == '__main__':
    main()