import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utilscripts.customDataset import *
# from papers_data_augmentations import *
from utilscripts.logger import create_logger
# from logger import * # import the logger functions
import timm # for using the XceptionNet model (pretrained)
# pip install timm # install the timm package to use the XceptionNet model
# from papers_data_augmentations import *
# import fornet
# from fornet import *
import random
# from train_val_test_v2 import test_one_epoch
# from gotcha_trn_val_tst import gotcha_test
from utilscripts.train_val_test import *



def test_model(model, pretrained_model_path, test_dataloader, dataset_labels, test_dir, model_name, exp_results_path, gotcha, device, args):
    # Evaluate the model on the test data
    if not gotcha:

        test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold = test_one_epoch(
            model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh
        )

        # print(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
        print(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
        print("Dataset labels: ", dataset_labels)
        print(f"Dataset test dir: {test_dir}")
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

        # # log results to wandb
        # if args.wandb:
        #     run.tags = run.tags + ('Testing',) # add testing tag
            
        #     run.log({'Test Accuracy': test_accuracy})
        #     run.log({'Balanced Test Accuracy': balanced_test_acc})
        #     run.log({'Test Accuracy Original': test_accuracy_original}) 
        #     run.log({'Test Accuracy SimSwap': test_accuracy_simswap}) 
        #     run.log({'Test Accuracy Ghost': test_accuracy_ghost})
        #     run.log({'Test Accuracy FaceDancer': test_accuracy_facedancer})
            
        #     run.log({'AP': ap_score})
        #     run.log({'TPR': TPR})
        #     run.log({'TNR': TNR})
        #     run.log({'AUC': auc_score})
        #     run.log({'EER': eer})
            

        # log the results to a log file
        if args.save_log:
            log_test_info_fows(args, pretrained_model_path, dataset_labels, exp_results_path, test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold)

    else: 
        print("Testing GOTCHA dataset!")

        # test_accuracy, balanced_test_accuracy, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, labels_collection, prediction_collection, img_path_collection = gotcha_test(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh) 
        
        # test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, \
        # TPR, TNR, auc_score, ap_score, eer, labels_list, img_path_collection, prob_original, \
        # prob_dfl, prob_fsgan, auc_best_threshold, eer_threshold 
        #img_path_collection, labels_collection, prediction_collection, 
        test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold = gotcha_test(model, test_dataloader, device, model_name, exp_results_path, args.tags, args.thresh)
        
        # print(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
        print(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
        print("Dataset labels: ", dataset_labels)
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
            log_test_info_gotcha(args, pretrained_model_path, exp_results_path, dataset_labels, test_dir, test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold)
            # print(f"logging results - {args.tags}")
            # # set up logger
            # log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
            # log = create_logger(log_path)
            # # log.info("")
            # # log.info(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
            # log.info(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
            # log.info(f"Model weights: {pretrained_model_path}")
            # log.info(f"Dataset labels: {dataset_labels}")
            # log.info(f"Dataset test dir: {test_dir}")
            # log.info(f"Data augmentation: {args.data_aug}")
            # log.info(f"Test Accuracy: {test_accuracy:.4f}")
            # log.info(f"Balanced Test Accuracy: {balanced_test_acc:.2f}")
            # log.info(f"Test Accuracy Original: {test_accuracy_original:.4f}")
            # log.info(f"Test Accuracy DFL: {test_accuracy_dfl:.4f}")
            # log.info(f"Test Accuracy FSGAN: {test_accuracy_fsgan:.4f}")
            
            # # log.info(f"AUC: {auc:.4f}")
            # log.info(f"Average Precision: {ap_score:.4f}")
            # log.info(f"TPR (recall): {TPR:.4f}")
            # log.info(f"TNR (specificity): {TNR:.4f}")
            # log.info(f"AUC: {auc_score:.4f}")
            # log.info(f"EER: {eer:.4f}")
            # log.info("Note the labels/preds/imgs_paths are in the json file")
            # log.info("")


def get_test_dataset(args):
    gotcha = False
    if args.dataset == 'milan_occ':
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

    else:
        print("Dataset not supported")
        exit()

    return test_dir, gotcha

def get_test_dataloader(args, test_dir):
    if args.data_aug == 'default':
        print("Using default data augmentation (thesis)")

        # Transformations for testing data
        test_transform = transforms.Compose([
            # resize image to 256x256
            transforms.Resize((256,256)), 
            # extract the center 224x224 part of the image -> makes the net concentrate on the face
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # test dataloader 
        test_dataset = FaceImagesDataset(test_dir, test_transform)
        # get the test_dataset labels
        dataset_labels = test_dataset.label_map
        print(dataset_labels)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size, #64, 
                                    sampler=sampler_test,
                                    num_workers = 3) # shuffle=False)

    elif args.data_aug == 'milan':
        print("using data augmentation from Milan paper")
        # train_transform = milan_train_transf()
        test_transform = milan_test_transf()


        # test dataloader 
        # test_dataset = FaceImagesAlbu(test_dir, test_transform)
        test_dataset = FaceImagesDataset(test_dir, test_transform, use_albu=True)

        # get the test_dataset labels
        dataset_labels = test_dataset.label_map
        print(dataset_labels)

        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size,   # 64
                                    sampler=sampler_test, 
                                    num_workers = 3)
        # shuffle=False)

    elif args.data_aug == 'median_filter':
        if 'milan' in args.dataset:
            print("Robustenss test data augmentation (Milan)")
            # test_transform = milan_test_transf(robusteness = True)
            test_transform = get_milan_test_robust_transform(robust_aug='median_filter')

        else: 
            print("Robustenss test data augmentation (thesis)")
        # ------------------------------------- #
        # Robusteness testing data augmentation #
        # - Median Filter
        # - Jpeg compression
        # - Change in brightness 
        # ------------------------------------- #
        # Transformations for testing data
        # https://albumentations.ai/docs/api_reference/full_reference/?h=median#albumentations.augmentations.blur.transforms.MedianBlur
        # https://albumentations.ai/docs/api_reference/full_reference/?h=jpeg#albumentations.augmentations.transforms.JpegCompression
        # https://albumentations.ai/docs/api_reference/full_reference/?h=brightness#albumentations.augmentations.transforms.RandomBrightnessContrast
        # NOTE: by default the probability of the transformations is set to 0.5 
            test_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.MedianBlur(blur_limit=(3,7), p = 1.0),  # Apply median filtering (remove noise)
                # A.ImageCompression(quality_lower=50, quality_upper=99, compression_type=0),  # Apply JPEG compression (add image compression artifacts)
                # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),  # Change in brightness and not contrast (set to 0)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
       
        # ------------------------------------------------------------------------------------------------------------------------------------------- #
        # test dataloader 
        test_dataset = FaceImagesDataset(test_dir, test_transform, use_albu=True)
        # get the test_dataset labels
        dataset_labels = test_dataset.label_map
        print(dataset_labels)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size, #64, 
                                    sampler=sampler_test,
                                    num_workers = 3) # shuffle=False)
        
    elif args.data_aug == 'jpeg_compression':
        if 'milan' in args.dataset:
            print("Robustenss test data augmentation (Milan)")
            # test_transform = milan_test_transf(robusteness = True)
            test_transform = get_milan_test_robust_transform(robust_aug='jpeg_compression')
        else:
            print("Robustenss test data augmentation (thesis)")
            test_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                # A.MedianBlur(blur_limit=(3,7)),  # Apply median filtering (remove noise)
                A.ImageCompression(quality_lower=50, quality_upper=99, compression_type=0, p=1.0),  # Apply JPEG compression (add image compression artifacts)
                # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),  # Change in brightness and not contrast (set to 0)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
       
        # ------------------------------------------------------------------------------------------------------------------------------------------- #
        # test dataloader 
        test_dataset = FaceImagesDataset(test_dir, test_transform, use_albu=True)
        # get the test_dataset labels
        dataset_labels = test_dataset.label_map
        print(dataset_labels)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size, #64, 
                                    sampler=sampler_test,
                                    num_workers = 3) # shuffle=False)
    
    elif args.data_aug == 'random_brightness':
        if 'milan' in args.dataset:
            print("Robustenss test data augmentation (Milan)")
            # test_transform = milan_test_transf(robusteness = True)
            test_transform = get_milan_test_robust_transform(robust_aug='random_brightness')
        else:
            print("Robustenss test data augmentation (thesis)")
            test_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                # A.MedianBlur(blur_limit=(3,7)),  # Apply median filtering (remove noise)
                # A.ImageCompression(quality_lower=50, quality_upper=99, compression_type=0, p=1.0),  # Apply JPEG compression (add image compression artifacts)
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),  # Change in brightness and not contrast (set to 0)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
       
        # ------------------------------------------------------------------------------------------------------------------------------------------- #
        # test dataloader 
        test_dataset = FaceImagesDataset(test_dir, test_transform, use_albu=True)
        # get the test_dataset labels
        dataset_labels = test_dataset.label_map
        print(dataset_labels)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size= args.batch_size, #64, 
                                    sampler=sampler_test,
                                    num_workers = 3) # shuffle=False)


    else:
        print("Data augmentation not supported")
        exit()

    # return dataset_labels,test_dataloader
    return test_dataloader, dataset_labels

# ------------------------------------------------------------------------------------------- #

def log_test_info_fows(args, pretrained_model_path, dataset_labels, exp_results_path, test_accuracy, balanced_test_accuracy, test_accuracy_original, TPR, TNR, auc_score, ap_score, eer, labels_collection, prediction_collection, img_path_collection, test_accuracy_dfl, test_accuracy_fsgan):
    print(f"logging results - {args.tags}")
            # set up logger
    log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
    log = create_logger(log_path)
            # log.info("")
            # log.info(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
    log.info(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
    log.info(f"Model weights: {pretrained_model_path}")
    log.info(f"Dataset labels: {dataset_labels}")
    log.info(f"Data augmentation: {args.data_aug}")
    log.info(f"Test Accuracy: {test_accuracy:.4f}")
    log.info(f"Balanced Test Accuracy: {balanced_test_accuracy:.2f}")
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
    log.info(f"Labels collection: {labels_collection}")
    log.info(f"Prediction collection: {prediction_collection}")
    log.info(f"Imgs paths collection: {img_path_collection}")
    log.info("")# end the wandb session


def log_test_info_gotcha(args, pretrained_model_path, exp_results_path, dataset_labels, test_dir, test_accuracy, balanced_test_acc, test_accuracy_original, test_accuracy_dfl, test_accuracy_fsgan, TPR, TNR, auc_score, ap_score, eer, auc_best_threshold, eer_threshold):
    print(f"logging results - {args.tags}")
    # set up logger
    log_path = log_path = exp_results_path+ f'/exp_logs/testing/output.log'
    log = create_logger(log_path)
    # log.info("")
    # log.info(f"Test results of the pretrained model {model_name} on the {args.dataset} dataset with thresh {args.thresh}")
    log.info(f"Test results of the pretrained model {args.model} on the {args.dataset} dataset with thresh {args.thresh}")
    log.info(f"Model weights: {pretrained_model_path}")
    log.info(f"Dataset labels: {dataset_labels}")
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

# ------------------------------------------------------------------------------------------- #

# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # Add arguments
    # model parameters
    parser.add_argument('--model', type=str, default='mnetv2', help='Model to use for training and testing')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    # parser.add_argument('--patience', type=int, default=3, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--thresh', type = float, default = 0.5, help = 'Threshold for binary classification.')
    # add flag to save logs to a file in the model folder
    parser.add_argument('--save-log', action='store_false', help='Save the model output logs')
    # parser.add_arugment('--save-model', action='store_true', help='Save the model') # used to save the model at the end of training
    # path to the model folder -> to be updated based on the model name and tags
    parser.add_argument('--save-model-path', action= "store_false", help='Use the saved pre-trained version of the model')

    parser.add_argument('--tl', action = 'store_true', help='Use the re-trained version of the model (transf learning)')
    parser.add_argument('--ft', action = 'store_true', help='Fine-Tuning the model')
    parser.add_argument('--gen', action = 'store_true', help='Testing the generalization capabilites of the model')
    parser.add_argument('--robust', action = 'store_true', help='Testing the robusteness of the model to different data augmentations')
    parser.add_argument('--data-aug', type=str, default='default', help='Data augmentation to use for training and testing') 
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='thesis_occ', help='Path to the training and/or testing dataset')
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


